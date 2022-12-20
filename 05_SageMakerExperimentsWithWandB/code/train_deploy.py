import argparse
import logging
import os
import random
import sys
import json
import numpy as np
import subprocess
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
import wandb
wandb.login(key="WANDB_API_KEY") # Pass your W&B API key here
wandb.init(project="Prodject_Name") # Add your W&B project name 
    
# compute metrics function for binary classification
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def train(args):
    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    test_dataset = load_from_disk(args.test_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")
    
    # download model from model hub
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # define training args
    training_args = TrainingArguments(
        output_dir=args.checkpoints,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        logging_dir=f"{args.checkpoints}/logs",
        learning_rate=args.learning_rate,
        report_to="wandb"
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    # train model
    trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.checkpoints, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    # Saves the model locally. In SageMaker, writing in /opt/ml/model sends it to S3
    trainer.save_model(args.model_dir)
    

# Inference related functions
def model_fn(model_dir):
    """
    Load the model for inference
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(model_dir, 'model/')
    # Load BERT tokenizer from disk.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    test_args = TrainingArguments(
        output_dir=model_path,
        do_train=False,
        do_predict=True,
        per_device_eval_batch_size=32,
        dataloader_drop_last=False
    )

    trainer = Trainer(
        model=model.to(device),
        args=test_args,
        compute_metrics=compute_metrics)

    model_dict = {'trainer': trainer, 'tokenizer': tokenizer}
    return model_dict


def predict_fn(input_data, model_dict):
    """
    Apply model to the incoming request
    """
    trainer = model_dict['trainer']
    tokenizer = model_dict['tokenizer']

    try:
        data = {"success": False}
        input_text = input_data['text']
        
        encoding = tokenizer(input_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        encoding = {k: v.to(trainer.model.device) for k, v in encoding.items()}

        outputs = trainer.model(**encoding)
        logits = outputs.logits

        # get probabilities for each topic
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= 0.5)] = 1
        probs = probs.cpu().detach().numpy()
        
        predictions = predictions.reshape(1, -1)
        probs = probs.reshape(1, -1)
        
        if probs[0][0] > probs[0][1]:
            results = {"label": "Negative", "score": probs[0][0]}
        else:
            results = {"label": "Positive", "score": probs[0][1]}

        return results

    except Exception as e:
        data = {"success": False}
        data['sentiments'] = 'NULL'
        print("Unexpected error in predict_fn:", e)
        return json.dumps(data)


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """
    if request_content_type == "application/json":
        try:
            request = json.loads(request_body)
            return request
        except:
            return request_body
    else:
        request = request_body
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=float, default=5e-5)

    # Data, model, and output directories
    parser.add_argument("--checkpoints", type=str, default="/opt/ml/checkpoints/")
    parser.add_argument("--model_dir", type=str, default=os.path.join(os.environ["SM_MODEL_DIR"], 'model/'))
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    train(parser.parse_args())    
