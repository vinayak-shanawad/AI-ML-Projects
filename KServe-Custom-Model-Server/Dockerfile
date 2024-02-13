# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.8-slim

ENV APP_HOME /app
WORKDIR $APP_HOME

# Install production dependencies.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r ./requirements.txt

# Copy local code to container image
COPY KServe_BERT_Sentiment_ModelServer.py ./

CMD ["python", "KServe_BERT_Sentiment_ModelServer.py"]