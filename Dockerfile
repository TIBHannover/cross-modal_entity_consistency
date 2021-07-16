FROM tensorflow/tensorflow:1.15.2-gpu-py3
MAINTAINER TIB-Visual-Analytics

RUN DEBIAN_FRONTEND=noninteractive apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install libsm6 libxext6 libxrender-dev -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install python-opencv -y
RUN pip install --upgrade pip

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm
RUN python -m spacy download de_core_news_sm 
