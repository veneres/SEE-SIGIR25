FROM nvcr.io/nvidia/pytorch:23.03-py3
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install default-jdk -y
RUN apt-get install swig -y
RUN python --version
RUN python -m pip install --upgrade pip
RUN pip install transformers datasets
RUN pip install ir_datasets ir_measures

WORKDIR /code

ARG UID=1000
ARG UNAME=testuser
RUN useradd -u $UID -m $UNAME
USER $UNAME


ENV IR_DATASETS_HOME="/ir_datasets"
ENV TRANSFORMERS_CACHE="/hf"
