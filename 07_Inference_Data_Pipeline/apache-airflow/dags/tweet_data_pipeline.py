from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
import pandas as pd
import os
import boto3
import csv
import json
import time


def download_tweets_dataset(local_file_path: str, s3_bucket: str, s3_key: str):
   # Create an S3 client with the access key and secret key
    s3 = boto3.client('s3')

    if not os.path.exists(local_file_path):
        os.makedirs(local_file_path)

    # Download the file from S3
    s3.download_file(s3_bucket, s3_key, os.path.join(local_file_path, 'tweet_data.csv'))

    print(f'{s3_bucket}/{s3_key} downloaded to {local_file_path}')


def clean_tweets_dataset(local_file_path: str):

    csv_file = os.path.join(local_file_path, 'tweet_data.csv')
    json_file = os.path.join(local_file_path, 'tweet_data.json')

    with open(csv_file, "r+") as infile, open(json_file, "w+") as outfile:
        reader = csv.reader(infile)
        for row in reader:
            data = [row[0].replace("@","")]
            outfile.write(json.dumps(data))
            outfile.write('\r\n')


def prepare_batch_request(local_file_path: str, s3_bucket: str, s3_key: str):
   # Create an S3 client with the access key and secret key
    s3 = boto3.client('s3')

    # Set the path to your local file
    local_file_path = os.path.join(local_file_path, 'tweet_data.json')

    # Upload the file to S3
    s3.upload_file(local_file_path, s3_bucket, s3_key)

    print(f'{local_file_path} uploaded to {s3_bucket}/{s3_key}')


def trigger_tweets_batch_inference_job(batch_input: str, batch_output: str, model_name):
    sm_boto3 = boto3.client('sagemaker', region_name='ap-south-1')
    
    batch_job_name = "tweets-batch-inference-job-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

    request = {
        "ModelClientConfig": {
            "InvocationsTimeoutInSeconds": 3600,
            "InvocationsMaxRetries": 3,
        },
        "TransformJobName": batch_job_name,
        "ModelName": model_name,
        "MaxPayloadInMB": 1,
        "BatchStrategy": "SingleRecord",
        "TransformOutput": {
            "S3OutputPath": batch_output,
            "AssembleWith": "Line",
            "Accept": "application/json",
        },
        "TransformInput": {
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": batch_input}},
            "SplitType": "Line",
            "ContentType": "application/json",
        },
        "TransformResources": {"InstanceType": "ml.c5.9xlarge", "InstanceCount": 1},
    }
    
    response = sm_boto3.create_transform_job(**request)
    print("response:", response)


def check_batch_inference_job_status(queue_name: str, **kwargs):
   # Create an S3 client with the access key and secret key
    sqs = boto3.client('sqs', region_name='xx-xxxxx-x')

    # Get the URL of the SQS queue
    response = sqs.get_queue_url(QueueName=queue_name)
    queue_url = response['QueueUrl']

    while True:
        # Read a message from the queue
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1
        )
        if 'Messages' in response:
            message = json.loads(response['Messages'][0]['Body'])
            print(message)

            # Fetch s3_output_path from message queue
            s3_output_path = message["detail"]["TransformOutput"]["S3OutputPath"]

            print(s3_output_path)

            # Delete the message from the queue after processing
            response = sqs.delete_message(
                QueueUrl=queue_url,
                ReceiptHandle=response['Messages'][0]['ReceiptHandle']
            )

            print(response)

            break
        else:
            print("No messages in the queue.")

        # Wait time
        time.sleep(30)

    task_instance = kwargs['task_instance']
    task_instance.xcom_push(key='s3_output_path_key', value=s3_output_path)


def download_batch_inference_results(local_dir: str, **kwargs):
   # Create an S3 client with the access key and secret key
    s3_client = boto3.client('s3')

    task_instance = kwargs['task_instance']
    s3_output_path = task_instance.xcom_pull(task_ids='Check_Tweets_Batch_Inference_Job_Status', key='s3_output_path_key')

    print(s3_output_path)

    s3_bucket, s3_key = s3_output_path.split('/')[2], '/'.join(s3_output_path.split('/')[3:])

    response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_key)
    print("response", response)

    # iterate over all objects in the S3 path
    i = 0
    for s3_object in response.get('Contents', []):
        i=i+1
        if i == 1:
            continue
        s3_key = s3_object['Key']
        local_path = os.path.join(local_dir, os.path.basename(s3_key))
        s3_client.download_file(s3_bucket, s3_key, local_path)
        print(f'Downloaded {s3_key} to {local_path}')
        


with DAG(
	dag_id="Tweets_Data_Pipeline",
	start_date=datetime(2023, 3, 22),
	schedule_interval=timedelta(days=1),
	catchup=False,
) as dag:

    start_task = DummyOperator(task_id='start')
    
    download_tweets_dataset = PythonOperator(
        task_id='Download_Tweets_Dataset',
        python_callable=download_tweets_dataset,
        op_kwargs={
            'local_file_path': '/home/ubuntu/airflow/data',
            's3_bucket': 'sagemaker-xx-xxxx-x-xxxxxxxxxxxx',
            's3_key': 'sagemaker/social-media/dataset/tweet_data.csv'
        }
    )

    clean_tweets_dataset = PythonOperator(
        task_id='Clean_Tweets_Dataset',
        python_callable=clean_tweets_dataset,
        op_kwargs={
            'local_file_path': '/home/ubuntu/airflow/data'
        }
    )

    prepare_batch_request = PythonOperator(
        task_id='Prepare_Batch_Request',
        python_callable=prepare_batch_request,
        op_kwargs={
            'local_file_path': '/home/ubuntu/airflow/data',
            's3_bucket': 'sagemaker-xx-xxxx-x-xxxxxxxxxxxx',
            's3_key': 'sagemaker/social-media/batch_transform/input/tweet_data.json'
        }
    )

    trigger_tweets_batch_inference_job = PythonOperator(
        task_id='Trigger_Tweets_Batch_Inference_Job',
        python_callable=trigger_tweets_batch_inference_job,
        op_kwargs={
            'batch_input': 's3://sagemaker-xx-xxxx-x-xxxxxxxxxxxx/sagemaker/social-media/batch_transform/input',
            'batch_output': 's3://sagemaker-xx-xxxx-x-xxxxxxxxxxxx/sagemaker/social-media/batch_transform/output',
            'model_name': 'tweet-classifier-v1-model-2023-03-22-12-10-53'
        }
    )

    check_batch_inference_job_status = PythonOperator(
        task_id='Check_Tweets_Batch_Inference_Job_Status',
        python_callable=check_batch_inference_job_status,
        op_kwargs={
            'queue_name': 'tweets-batch-inference-job-queue'
        }
    )

    download_batch_inference_results = PythonOperator(
        task_id='Download_Batch_Inference_Results',
        python_callable=download_batch_inference_results,
        op_kwargs={
            'local_dir': '/home/ubuntu/airflow/data'
        }
    )

    end_task = DummyOperator(task_id='end')

start_task >> download_tweets_dataset >> clean_tweets_dataset >> prepare_batch_request >> trigger_tweets_batch_inference_job >> end_task

start_task >> check_batch_inference_job_status >> download_batch_inference_results >> end_task