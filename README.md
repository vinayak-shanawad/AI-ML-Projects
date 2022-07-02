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

### 2. Multi-Model Endpoints with Hugging Face Transformers and Amazon SageMaker

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
 
