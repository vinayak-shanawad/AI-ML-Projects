# AWS SageMaker Projects


### 1. AWS SageMaker: Train, Deploy and Update a HuggingFace BERT model on Disaster Tweets Classification dataset

Text classification is a technique for putting text into different categories and has a wide range of applications: email providers use text classification to  detect to spam emails, marketing agencies use it for sentiment analysis of customer reviews, and moderators of discussion forums use it to detect inappropriate comments.

Twitter has become an important communication channel in times of emergency. [Kaggle competition dataset](https://www.kaggle.com/c/nlp-getting-started/overview), which consists of fake and real Tweets about disasters. The task is to classify the tweets.

We covered below steps in this project.
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
