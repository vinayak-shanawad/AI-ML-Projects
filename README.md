# AWS SageMaker Projects


### 1. AWS SageMaker: Train, Deploy and Update a HuggingFace BERT model on Disaster Tweets Classification dataset

Text classification is a technique for putting text into different categories and has a wide range of applications: email providers use text classification to  detect spam emails, marketing agencies use it for sentiment analysis of customer reviews, and moderators of discussion forums use it to detect inappropriate comments.

Twitter has become an important communication channel in times of emergency. [Kaggle competition dataset](https://www.kaggle.com/c/nlp-getting-started/overview), which consists of fake and real Tweets about disasters. The task is to classify the tweets.

We covered the steps below in this project.
- Setup
- Data Preparation
- EDA
- Amazon SageMaker Training
- Train on Amazon SageMaker using on-demand instances with Epoch=2
- Train on Amazon SageMaker using spot instances
- Host the model on an Amazon SageMaker Endpoint
- Train on Amazon SageMaker using on-demand instances with Epoch=3
- Update a SageMaker model endpoint
- Cleanup

Please refer to the [Medium article](https://medium.com/@vinayakshanawad/aws-sagemaker-train-deploy-and-update-a-hugging-face-bert-model-eeefc8211368) for detailed information.

### 2. Bring Your BERT Model With Amazon SageMaker Script Mode

[Script mode](https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-script-mode/index.html) allows you to write custom training and inference script against commonly used ML framework containers maintained by AWS. Script mode is easy to use and extremely flexible.

We covered the steps below in this project.
- Development Environment and Permissions
- Store Model Artifacts
- Write the Inference Script
- Package Model
- Upload Hugging Face model to S3
- Create SageMaker Real-time endpoint
- Get Predictions
- Update SageMaker Real-time endpoint
- Delete the Real-time endpoint

Please refer to the [Medium article](https://medium.com/@vinayakshanawad/bring-your-own-model-with-amazon-sagemaker-script-mode-6cf374747f9e) for detailed information.

### 3. Multi-Model Endpoints with Hugging Face Transformers and Amazon SageMaker

With Amazon SageMaker multi-model endpoints, customers can create an endpoint that seamlessly hosts up to thousands of models. These endpoints are well suited to use cases where any one of many models, which can be served from a common inference container, needs to be callable on-demand and where it is acceptable for infrequently invoked models to incur some additional latency.

We covered the steps below in this project.
- Development Environment and Permissions
- Retrieve Model Artifacts
- Write the Inference Script
- Package Models
- Upload multiple Hugging Face models to S3
- Create Multi-Model Endpoint
- Get Predictions
- Dynamically deploying models and Updating a model to the endpoint
- Delete the Multi-Model Endpoint

Please refer to the [Medium article](https://medium.com/@vinayakshanawad/multi-model-endpoints-with-hugging-face-transformers-and-amazon-sagemaker-c0e5a3693fac) for detailed information.

### 4. Pay as you use SageMaker Serverless inference with GPT-2

SageMaker (SM) Serverless inference option allows you to focus on the model building process without having to manage the underlying infrastructure. You can choose either a SM in-built container or bring your own.

We covered the steps below in this project.
- SageMaker Serverless inference Use cases
- Warming up the Cold Starts
- Serverless Inference example
- Monitor Serverless GPT-2 model endpoint

Please refer to the [Medium article](https://medium.com/@vinayakshanawad/pay-as-you-use-sagemaker-serverless-inference-with-gpt-2-62b23485f828) for detailed information.
 
### 5. AWS SageMaker Experiments with Weights and Biases

Amazon SageMaker experiments to organize, track, compare and evaluate our machine learning experiments on IMDB movie reviews dataset then deploy the endpoint for best training job or trial component.

We covered the steps below in this project.
- Set up the experiment
- Track experiment
- Accessing Training Metrics using Experiments UI from SageMaker Studio
- Accessing Training Metrics using SageMaker TrainingJobAnalytics API
- Accessing Training Metrics using Weights and Biases
- Compare the model training runs for an experiment
- Deploy endpoint for the best training-job or trial component

Please refer to the [Medium article](https://medium.com/@vinayakshanawad/aws-sagemaker-experiments-with-weights-and-biases-ea0932658a4f) for detailed information.

### 6. Shadow deployment of ML models with Amazon SageMaker

AWS has announced the shadow model deployment strategy support in Amazon SageMaker in `AWS re:Invent 2022`. Shadow testing helps us to minimize the risk of deploying a low performing model, minimize the downtime and monitor the model performance of the new model version for a period of time and can rollback if there is an issue with the new version.

We covered the steps below in this project.
- Deploy tweet-classifier-v1 model (as production variant
- Get predictions from tweet-classifier-v1 model
- Deploy tweet-classifier-v2 model (as shadow variant)
- Get predictions from tweet-classifier-v2 model
- View production variant captured data from S3
- View shadow variant captured data from S3
- Compare the model evaluation metrics
- Promote the shadow variant as a production variant

Please refer to the [Medium article](https://medium.com/@vinayakshanawad/shadow-deployment-of-ml-models-with-amazon-sagemaker-65e6816821ae) for detailed information.

### 7. ML Inference Data Pipeline using SageMaker and Airflow

Accelerate and automate ML inference data pipeline using SageMaker and Airflow.

We covered the steps below in this project.
- Real-world batch inference use cases
- Create tweet-classifier-v1 model
- Build a Tweets inference data pipeline
- Monitor Airflow DAG and it's workflow execution

Please refer to the [Medium article](https://medium.com/@vinayakshanawad/accelerate-ml-inference-data-pipeline-using-sagemaker-and-apache-airflow-f19207e896ca) for detailed information.

### 8. Monitoring and Saving SageMaker Inference Expenses

Tips and Tools for Effective Monitoring and Savings.

We covered the steps below in this post.
- Retrieve endpoint and its instance details
- Compute endpoint age in days
- Comput endpoint total invocations and invocations count in last 15 days
- Compute instance cost details in dollors

Please refer to the [Medium article](https://medium.com/@vinayakshanawad/monitoring-and-saving-sagemaker-inference-expenses-f6795a9193ab) for detailed information.

### 9. Supercharge BERT Inference with AWS Inferentia2 and Hugging Face Transformers

Achieve 2–3ms inference speed and high throughput for Text Classification tasks

We covered the steps below in this post.
- Convert your Hugging Face Transformer to AWS Neuron (Inferentia2)
- Create a custom inference.py script for text-classification
- Create and upload the neuron model and inference script to Amazon S3
- Deploy a Real-time Inference Endpoint on Amazon SageMaker
- Run and evaluate Inference performance of BERT on Inferentia2
- Clean Up

Please refer to the [Medium article](https://medium.com/@vinayakshanawad/say-goodbye-to-costly-bert-inference-turbocharge-with-aws-inferentia2-and-hugging-face-c30a21df6b4e) for detailed information.