# This file is automatically generated by pipeline-ai and should not be changed by hand.
FROM python:3.10-slim

WORKDIR /app

RUN apt update -y
RUN pip install -U pip setuptools wheel

# Install serving packages
RUN pip install -U fastapi==0.105.0 uvicorn==0.25.0 \
    python-multipart==0.0.6 loguru==0.7.2

# Container commands
RUN apt-get update 
RUN apt-get install -y git 


# Install python dependencies
RUN pip install pipeline-ai torch transformers

# Copy in files
COPY ./ ./

ENV PIPELINE_PATH=large-v2-detect-new:my_pl
ENV PIPELINE_NAME=uriel/insane-whisper-v2
ENV PIPELINE_IMAGE=uriel/insane-whisper-v2

CMD ["uvicorn", "pipeline.container.startup:create_app", "--host", "0.0.0.0", "--port", "14300"]
