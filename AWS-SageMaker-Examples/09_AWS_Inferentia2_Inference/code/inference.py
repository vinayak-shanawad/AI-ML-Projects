
import os
import json
import torch
from transformers import AutoConfig, AutoTokenizer

# To use one neuron core per worker
os.environ["NEURON_RT_NUM_CORES"] = "1"

def model_fn(model_dir):
    """
    Load the model for inference
    """   
    # Load tokenizer and neuron model from model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    neuron_model = torch.jit.load(os.path.join(model_dir, "model.neuron"))
    model_config = AutoConfig.from_pretrained(model_dir)
    model_dict = {'neuron_model': neuron_model, 'tokenizer': tokenizer, 'model_config': model_config}
    
    return model_dict

def predict_fn(input_data, model):
    """Apply model to the incoming request.
    
    Documents 
    """
    tokenizer = model['tokenizer']
    neuron_model = model['neuron_model']
    model_config = model['model_config']

    sequence = input_data["sequence"]

    embeddings = tokenizer(
        sequence,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True,
    )
    
    # convert to tuple for neuron model
    neuron_inputs = tuple(embeddings.values())
    
    # run prediciton
    with torch.no_grad():
        predictions = neuron_model(*neuron_inputs)[0]
        scores = torch.nn.Softmax(dim=1)(predictions)

    # return dictonary, which will be json serializable
    return [{"label": model_config.id2label[item.argmax().item()], "score": item.max().item()} for item in scores]
