import json, datetime
import boto3
import openai
import logging
import pandas
import os
import tscribe

from urllib.parse import urlparse 

openai.api_key = os.environ['OpenAIKey']

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def convert_time_stamp(timestamp: str) -> str:
    """ Function to help convert timestamps from s to H:M:S """
    ## Taken from `tscribe`; can't get `import tscribe` to provide 
    # the required namespace, and debugging why is causing headaches. 
    # Everything else works!

    delta = datetime.timedelta(seconds=float(timestamp))
    seconds = delta - datetime.timedelta(microseconds=delta.microseconds)
    return str(seconds)

def decode_transcript_to_dataframe(data: str):
    """Decode the transcript into a pandas dataframe"""
    ## Taken from `tscribe`; can't get `import tscribe` to provide 
    # the required namespace, and debugging why is causing headaches. 
    # Everything else works!

    logging.info("Decoding transcript")

    decoded_data = {"start_time": [], "end_time": [], "speaker": [], "comment": []}

    # If speaker identification
    if "speaker_labels" in data["results"].keys():
        logging.debug("Transcipt has speaker_labels")

        # A segment is a blob of pronounciation and punctuation by an individual speaker
        for segment in data["results"]["speaker_labels"]["segments"]:

            # If there is content in the segment, add a row, write the time and speaker
            if len(segment["items"]) > 0:
                decoded_data["start_time"].append(
                    convert_time_stamp(segment["start_time"])
                )
                decoded_data["end_time"].append(convert_time_stamp(segment["end_time"]))
                decoded_data["speaker"].append(segment["speaker_label"])
                decoded_data["comment"].append("")

                # For each word in the segment...
                for word in segment["items"]:

                    # Get the word with the highest confidence
                    pronunciations = list(
                        filter(
                            lambda x: x["type"] == "pronunciation",
                            data["results"]["items"],
                        )
                    )
                    word_result = list(
                        filter(
                            lambda x: x["start_time"] == word["start_time"]
                            and x["end_time"] == word["end_time"],
                            pronunciations,
                        )
                    )
                    result = sorted(
                        word_result[-1]["alternatives"], key=lambda x: x["confidence"]
                    )[-1]

                    # Write the word
                    decoded_data["comment"][-1] += " " + result["content"]

                    # If the next item is punctuation, write it
                    try:
                        word_result_index = data["results"]["items"].index(
                            word_result[0]
                        )
                        next_item = data["results"]["items"][word_result_index + 1]
                        if next_item["type"] == "punctuation":
                            decoded_data["comment"][-1] += next_item["alternatives"][0][
                                "content"
                            ]
                    except IndexError:
                        pass

    # If channel identification
    elif "channel_labels" in data["results"].keys():
        logging.debug("Transcipt has channel_labels")

        # For each word in the results
        for word in data["results"]["items"]:

            # Punctuation items do not include a start_time
            if "start_time" not in word.keys():
                continue

            # Identify the channel
            channel = list(
                filter(
                    lambda x: word in x["items"],
                    data["results"]["channel_labels"]["channels"],
                )
            )[0]["channel_label"]

            # If still on the same channel, add the current word to the line
            if (
                channel in decoded_data["speaker"]
                and decoded_data["speaker"][-1] == channel
            ):
                current_word = sorted(
                    word["alternatives"], key=lambda x: x["confidence"]
                )[-1]
                decoded_data["comment"][-1] += " " + current_word["content"]

            # Else start a new line
            else:
                decoded_data["start_time"].append(
                    convert_time_stamp(word["start_time"])
                )
                decoded_data["end_time"].append(convert_time_stamp(word["end_time"]))
                decoded_data["speaker"].append(channel)
                current_word = sorted(
                    word["alternatives"], key=lambda x: x["confidence"]
                )[-1]
                decoded_data["comment"].append(current_word["content"])

            # If the next item is punctuation, write it
            try:
                word_result_index = data["results"]["items"].index(word)
                next_item = data["results"]["items"][word_result_index + 1]
                if next_item["type"] == "punctuation":
                    decoded_data["comment"][-1] += next_item["alternatives"][0][
                        "content"
                    ]
            except IndexError:
                pass

    # Neither speaker nor channel identification
    else:
        logging.debug("No speaker_labels or channel_labels")

        decoded_data["start_time"] = convert_time_stamp(
            list(
                filter(lambda x: x["type"] == "pronunciation", data["results"]["items"])
            )[0]["start_time"]
        )
        decoded_data["end_time"] = convert_time_stamp(
            list(
                filter(lambda x: x["type"] == "pronunciation", data["results"]["items"])
            )[-1]["end_time"]
        )
        decoded_data["speaker"].append("")
        decoded_data["comment"].append("")

        # Add words
        for word in data["results"]["items"]:

            # Get the word with the highest confidence
            result = sorted(word["alternatives"], key=lambda x: x["confidence"])[-1]

            # Write the word
            decoded_data["comment"][-1] += " " + result["content"]

            # If the next item is punctuation, write it
            try:
                word_result_index = data["results"]["items"].index(word)
                next_item = data["results"]["items"][word_result_index + 1]
                if next_item["type"] == "punctuation":
                    decoded_data["comment"][-1] += next_item["alternatives"][0][
                        "content"
                    ]
            except IndexError:
                pass

    # Produce pandas dataframe
    dataframe = pandas.DataFrame(
        decoded_data, columns=["start_time", "end_time", "speaker", "comment"]
    )

    # Clean leading whitespace
    dataframe["comment"] = dataframe["comment"].str.lstrip()

    return dataframe

def count_tokens(input :str):
    logging.debug("Counting tokens")
    return len(input.split(" "))

def create_transcript(input: str, max_tokens, start_row=0):
    """
        This function takes and existing string with NL characters, 
        and delivers a new string that has at most 'max_tokens' words 
        in it. You can offset the starting line by using `start_row`.
    """
    logging.debug("IN create_transcript")
    lines = input.split('\n')
    count = start_row
    temp_prompt = ""
    while count_tokens(temp_prompt) < max_tokens and count < len(lines):
        temp_prompt += lines[count]+'\n'
        count+=1
    
    if count == len(lines):
        logging.debug("OUT create_transcript")
        return {'prompt':temp_prompt,'count':count,'total':len(lines)}
    
    logging.debug("Digging deeper")
    prompt = ""
    final_count = start_row
    while final_count < count:
        prompt+=lines[final_count]+'\n'
        final_count+=1

    logging.debug("OUT create_transcript")
    return {'prompt':prompt,'count':count,'total':len(lines)}
    

def lambda_handler(event, context):
    logging.info("Underpants!")
    logging.debug(event)

    bucket=event['Records'][0]['s3']['bucket']['name']
    key=event['Records'][0]['s3']['object']['key']

    # grab the transcript
    logging.debug("Making boto clients")
    s3 = boto3.client("s3", region_name="us-east-1") ## TODO may have to change this.>>
    sns = boto3.client("sns")
    
    logging.debug("Making S3 object for key")
    s3_object = s3.get_object(Bucket=bucket, Key=key)
    
    logging.debug("Loading contents of transcription into variable")
    transcript_job_result = json.loads(s3_object['Body'].read())

    logging.debug("Converting transcript into a dataframe")
    df = decode_transcript_to_dataframe(transcript_job_result)
        
    logging.debug("Converting dataframe into a string/transcript")
    full_transcript = df.to_string(header=True, index=False, columns=["speaker","comment"])

    prompt = "Summerize this meeting transcript into a list of talking points and actions for me to distribute to attendees: "

    ext_prompt = "Knowing the earlier summary, add additional talking points and actions to the summary below, based on this meeting transcript that follows: "

    logging.info("Setting GPT model")
    model="gpt-3.5-turbo"

    try:

        logging.info("Creating a tokenised transcript")
        response = create_transcript(full_transcript,4097 - len(prompt.split(" ")))

        if response['count']==response['total']:
            logging.info("Transcript fits into a single API call. Sending it now.")
            chat_completion = openai.ChatCompletion.create(
                model=model, 
                messages=[{"role": "user", "content": prompt+full_transcript}]
                )
            logging.info(chat_completion)
            summary = chat_completion.choices[0].message.content
        else:
            logging.info("Transcript DOES NOT fit into a single API call. Sending the first part.")
            chat_completion = openai.ChatCompletion.create(
                model=model, 
                messages=[{"role": "user", "content": prompt+response['prompt']}]
                )
            num_prompts = 0
            while response['count']<response['total'] and num_prompts <3:
                num_prompts+=1
                context = chat_completion.choices[0].message.content
                logging.info("Creating chunk number "+(num_prompts+1))
                response = create_transcript(full_transcript,4097 - len(ext_prompt.split(" ")) - len(context.split(" ")),start_row=response['count'])
                logging.info("Sending off this chunk to GPT")
                chat_completion = openai.ChatCompletion.create(
                    model=model, 
                    messages=[{"role": "user", "content": ext_prompt+context+'\n'+response['prompt']}]
                    )
                logging.info(chat_completion)
            logging.info("Done with that, ready to send the summary to SNS")    
            summary = chat_completion.choices[0].message.content
     
        # print the chat completion
        logging.info(summary)

        logging.info("Send the summary to the SNS topic")
        sns.publish(
            TopicArn=os.environ['SNSTopic'],
            Message=summary,
            Subject="Meeting Transcript")
    except Exception as e:
        logging.error(e)
    
    logging.info("Profit!")

    return {
        'statusCode': 200,
        'body': json.dumps('All done!')
    }