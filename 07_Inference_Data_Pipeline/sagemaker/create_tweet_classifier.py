import os
import sagemaker
import boto3
from time import gmtime, strftime

sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()
region = sagemaker_session.boto_session.region_name
sm_boto3 = boto3.client("sagemaker")


def name_with_timestamp(name):
    return '{}-{}'.format(name, strftime('%Y-%m-%d-%H-%M-%S', gmtime()))

tweet_cls_v1_model_name    = name_with_timestamp('tweet-classifier-v1-model')

model_v1_path = "s3://sagemaker-xx-xxxx-x-xxxxxxxxxxxx/sagemaker/social-media/models/model_v1/model_v1.tar.gz"

production_variant_name = "AllTraffic"

# Create a tweet-classifier-v1 model as production variant
from sagemaker import get_execution_role

image_uri = sagemaker.image_uris.retrieve(
    framework="pytorch",
    region=region,
    py_version="py38",
    image_scope="inference",
    version="1.9",
    instance_type="ml.m5.xlarge",
)

primary_container = {'Image': image_uri, 'ModelDataUrl': model_v1_path,
    'Environment': {
        'SAGEMAKER_PROGRAM': 'train_deploy.py',
        'SAGEMAKER_REGION': region,
        'SAGEMAKER_SUBMIT_DIRECTORY': model_v1_path
    }
}

create_model_response = sm_boto3.create_model(ModelName = tweet_cls_v1_model_name, ExecutionRoleArn = get_execution_role(), PrimaryContainer = primary_container)
print('ModelArn= {}'.format(create_model_response['ModelArn']))