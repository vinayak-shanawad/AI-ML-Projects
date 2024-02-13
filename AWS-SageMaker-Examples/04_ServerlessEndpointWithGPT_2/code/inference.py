import os
import json
from transformers import GPT2Tokenizer, TextGenerationPipeline, GPT2LMHeadModel

# Load the model for inference
def model_fn(model_dir):

    # Load GPT2 tokenizer from disk.
    vocab_path = os.path.join(model_dir, 'vocab.json')
    merges_path = os.path.join(model_dir, 'merges.txt')
    
    tokenizer = GPT2Tokenizer(vocab_file=vocab_path, merges_file=merges_path)

    # Load GPT2 model from disk.
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    return TextGenerationPipeline(model=model, tokenizer=tokenizer)

# Apply model to the incoming request
def predict_fn(input_data, model):
    return model.__call__(input_data)

# Deserialize and prepare the prediction input
def input_fn(request_body, request_content_type):

    if request_content_type == "application/json":
        request = json.loads(request_body)
    else:
        request = request_body

    return request

# Serialize and prepare the prediction output
def output_fn(prediction, response_content_type):
    return str(prediction)