import json, datetime
import boto3
import logging
import pandas
import os
import tscribe

from urllib.parse import urlparse 


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

def call_br(prompt="Hello World",modelId='anthropic.claude-v2'):

    client = boto3.client("bedrock-runtime",region_name='us-east-1')

    accept = 'application/json'
    contentType = 'application/json'

    fixed_prompt =  "\n\nHuman:{PROMPT}\n\nAssistant:".format(PROMPT=prompt)
    body = json.dumps({
        "prompt": fixed_prompt,
        "max_tokens_to_sample": 1000,
        "temperature": 0.1,
        "top_p": 0.9,
    })

    response = client.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())

    return(response_body.get('completion'))

def get_summary(transcript):
    return call_br("You are an executive assistant, and responsible for writing the meeting minutes. The minutes must include the most important information and decisions made, and a list of actions that were agreed. Where possible to derive from the transcript, you should name the speakers in your minutes. Only provide the summary; do not add any prefix or suffix to your response.\n\n<transcript>{TRANSCRIPT}</transcript>\n\n".format(TRANSCRIPT=transcript))

def get_title(summary):
    return call_br("Please use the following meeting summary to create a meeting title with fewer than 10 words. Only provide the meeting title; do not provide any prefix or suffix in your response.\n\n<summary>{SUMMARY}</summary>\n\n".format(SUMMARY=summary))

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

    summary=get_summary(full_transcript)

    title=get_title(summary)
     
    # print the chat completion
    logging.info(summary)

    logging.info("Send the summary to the SNS topic")
    sns.publish(
        TopicArn=os.environ['SNSTopic'],
        Message="SUMMARY\n-------\n{summary}\nACTIONS\n-------\n{actions}".format(summary=summary,actions=actions),
        Subject=title)
    
    logging.info("Profit!")

    return {
        'statusCode': 200,
        'body': json.dumps('All done!')
    }
