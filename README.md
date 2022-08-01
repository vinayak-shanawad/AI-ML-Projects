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
 
