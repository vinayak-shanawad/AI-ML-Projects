from typing import Dict 

import kserve
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from kserve import ModelServer
import logging

class KServeBERTSentimentModel(kserve.Model):

    def __init__(self, name: str):
        super().__init__(name)
        KSERVE_LOGGER_NAME = 'kserve'
        self.logger = logging.getLogger(KSERVE_LOGGER_NAME)
        self.name = name
        self.ready = False
        

    def load(self):
        # Build tokenizer and model
        name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, torchscript=True)
        self.ready = True


    def predict(self, request: Dict, headers: Dict) -> Dict:
        
        sequence = request["sequence"]
        self.logger.info(f"sequence:-- {sequence}")

        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True,
        )

        # run prediciton
        with torch.no_grad():
            predictions = self.model(**inputs)[0]
            scores = torch.nn.Softmax(dim=1)(predictions)

        results = [{"label": self.model.config.id2label[item.argmax().item()], "score": item.max().item()} for item in scores]
        self.logger.info(f"results:-- {results}")

        # return dictonary, which will be json serializable
        return {"predictions": results}
    

if __name__ == "__main__":

  model = KServeBERTSentimentModel("bert-sentiment")
  model.load()

  model_server = ModelServer(http_port=8080, workers=1)
  model_server.start([model])