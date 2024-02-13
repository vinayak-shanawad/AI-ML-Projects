# Deploying a HuggingFace BERT Model with Seldon Core V2

The objective of this project is to demonstrate how we can deploy custom NLP models on Seldon Core v2. The custom model I have is the uniqueness model defines a score between 0 to 1 which measures the level of uniqueness in the tweets. This model computes a single uniqueness score by taking a list of tweets as input.

We covered the steps below in this project.
- What is Seldon Core?
- Inference Servers in Core v2
- Deploying custom NLP models in Seldon Core V2
- Deploying custom NLP models to Kubernetes

Please refer to the [Medium article](https://medium.com/@vinayakshanawad/deploying-a-huggingface-bert-model-with-kserve-3e521d69e596) for detailed information