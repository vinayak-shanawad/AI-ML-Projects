import json
import logging
import os
import sys
import numpy as np
import torch
from transformers import BertForSequenceClassification, BertTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

MAX_LEN = 64  # this is the max length of the sentence

print("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(model_dir)
    print("================ model loaded ===========================")
    return model.to(device)


def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    class_label = {1: "Real disaster",
               0: "Not a disaster"}

    input_id, input_mask = input_data
    input_id = input_id.to(device)
    input_mask = input_mask.to(device)
    print("============== encoded data =================")
    print(input_id, input_mask)
    with torch.no_grad():
        y = model(input_id, attention_mask=input_mask)[0]
        result = list(np.argmax(y, axis=1))
        result = [int(l) for l in result]
        print("=============== inference result =================")

        predicted_labels = [class_label[l] for l in result]
        print(predicted_labels)
    return predicted_labels


def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled tensor"""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        print("================ input sentences ===============")
        print(data)
        
        if isinstance(data, str):
            data = [data]
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
            pass
        else:
            raise ValueError("Unsupported input type. Input type can be a string or an non-empty list. \
                             I got {}".format(data))
                       
        input_ids = [tokenizer.encode(x, add_special_tokens=True) for x in data]
        
        print("================ encoded sentences ==============")
        print(input_ids)

        # pad shorter sentence
        padded =  torch.zeros(len(input_ids), MAX_LEN) 
        for i, p in enumerate(input_ids):
            padded[i, :len(p)] = torch.tensor(p)
     
        # create mask
        mask = (padded != 0)
        
        print("================= padded input and attention mask ================")
        print(padded, '\n', mask)

        return padded.long(), mask.long()
    raise ValueError("Unsupported content type: {}".format(request_content_type))


def output_fn(prediction, response_content_type):
    """
    Serialize and prepare the prediction output
    """
    
    if response_content_type == "application/json":
        response = json.dumps(prediction)
    else:
        response = json.dumps(prediction)

    return response