import json
import boto3
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    logging.info("Underpants!")
    logging.debug(event)

    logging.debug("Creating Transcribe client")
    transcribe = boto3.client('transcribe')
    
    logging.debug("Creating File URI for recording")
    file_uri = "https://s3.amazonaws.com/{bucket}/{key}".format(bucket=event['Records'][0]['s3']['bucket']['name'],key=event['Records'][0]['s3']['object']['key'])
    logging.debug(file_uri)
    job_name = event['Records'][0]['s3']['object']['sequencer']
    logging.debug(job_name)

    logging.debug("Sending Job")
    try:
        response=transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': file_uri},
            OutputBucketName=event['Records'][0]['s3']['bucket']['name'],
            MediaFormat=os.path.splitext(event['Records'][0]['s3']['object']['key'])[1].replace(".",""),
            LanguageCode='en-US',
            Settings={"ShowSpeakerLabels": True, "MaxSpeakerLabels":8}
            )
        logging.info(response)
    except Exception as e:
        logging.error(e)

    logging.info("Profit")

    return {
        'statusCode': 200,
        'body': json.dumps('All done.')
    }


