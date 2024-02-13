from mlserver import MLModel, types
from mlserver.codecs import StringCodec
from mlserver.utils import get_model_uri
from typing import Dict, List, Any
import json
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
import unicodedata
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class TextUniqueness(MLModel):
    async def load(self) -> bool:
        model_uri = await get_model_uri(self._settings)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load SentenceTransformer from disk.
        self.model = SentenceTransformer(model_uri, device=self.device)
        self.ready = True
        return self.ready
    
    
    # Logic for making predictions against our model
    async def predict(self, payload: types.InferenceRequest) -> types.InferenceResponse:
        try:
            request = self._extract_json(payload).get("echo_request", {})
            tweets = request.get("tweets", [])

            # Preprocessing - clean tweets
            cleaned_tweets = [self.clean_tweet(tweet) for tweet in tweets]

            output_data = {"success": True}
            
            # Encode all sentences
            embeddings = self.model.encode(list(cleaned_tweets))
            
            # Compute cosine similarity between all pairs
            cos_sim = util.cos_sim(embeddings, embeddings)
            
            # Add all pairs to a list with their cosine similarity score
            all_sentence_combinations = []
            for i in range(len(cos_sim)-1):
                for j in range(i+1, len(cos_sim)):
                    all_sentence_combinations.append(abs(np.float64(cos_sim[i][j].numpy())))

            output_data["uniqueness_score"] = round(1-(np.mean(all_sentence_combinations)),6)

            # Postprocessing - define the uniqueness bracket
            output_data["uniqueness_bracket"] = self.defineBuckets(output_data["uniqueness_score"])

        except Exception as e:
            logger.error("Unexpected error: %s", str(e))
            output_data = dict(uniqueness_score='NULL', success=False, error=str(e), loc="inference.predict")

        response_bytes = json.dumps(output_data).encode("UTF-8")

        return types.InferenceResponse(
            id=payload.id,
            model_name=self.name,
            model_version=self.version,
            outputs=[
                types.ResponseOutput(
                    name="echo_response",
                    shape=[len(response_bytes)],
                    datatype="BYTES",
                    data=[response_bytes],
                    parameters=types.Parameters(content_type="str"),
                )
            ],
        )
    

    def _extract_json(self, payload: types.InferenceRequest) -> Dict[str, Any]:
        inputs = {}
        for inp in payload.inputs:
            inputs[inp.name] = json.loads(
                "".join(self.decode(inp, default_codec=StringCodec))
            )
        
        return inputs
    

    def clean_tweet(self, text) -> str:
        text = re.sub('(\s)@\w+|^@\w+', '', text) # removing handles from tweets
        text = re.sub("#[A-Za-z0-9_]+","", text) # removing hastags
        text = re.sub(r"http\S+", '', text) # remove URLs
        text = (unicodedata.normalize('NFKD', text)
                    .encode('ascii', 'ignore')
                    .decode('utf-8', 'ignore'))
        text = re.sub("'", "", text) # remove single quotes
        text = re.sub('"', "", text) # remove double quotes
        text = re.sub("[^a-zA-Z0-9 \.]", "", text) #
        text = re.sub(r'\.+', ".", text) # replace multiple full stops
        text = re.sub(' +', ' ', text) # remove more than one space
        text.lstrip('0123456789.- ').rstrip('0123456789.- ')
        text = text.strip('0123456789.- ') # remove numbers at the beginning of a tweet
        text = text.strip() # remove empty spaces from left and right
        return text


    def defineBuckets(self, score) -> str:
        '''
        Function to get the buckets for the uniqueness score
        '''
        if score>=0.80:
            bracket = 'Extreme'
        elif score>=0.60:
            bracket = 'High'
        elif score>=0.40:
            bracket = 'Moderate'
        elif score>=0.20:
            bracket = 'Neutral'
        else:
            bracket = 'Low' 
        return bracket