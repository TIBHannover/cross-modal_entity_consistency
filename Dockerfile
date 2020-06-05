FROM tensorflow/tensorflow:1.15.2-gpu-py3
MAINTAINER TIB-Visual-Analytics

RUN DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN pip install --upgrade pip
RUN pip install fasttext==0.9.2
RUN pip install h5py==2.10.0
RUN pip install newspaper3k==0.2.8
RUN pip install numpy==1.18.4
RUN pip install PyYAML==5.3.1
RUN pip install requests==2.23.0
RUN pip install scikit-learn==0.23.1
RUN pip install scipy==1.4.1
RUN pip install spacy==2.2.4
RUN pip install tqdm==4.46.0
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download de_core_news_sm
