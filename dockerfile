FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04

RUN apt update && \
    apt upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt install -y tzdata && \
    apt install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install -y python3.10 python3.10-distutils git curl openssh-client

RUN ssh -o StrictHostKeyChecking=accept-new earth.kplabs.pl || true

ENV PYTHONUNBUFFERED=1

ADD . /workspace/
WORKDIR /workspace/

RUN curl -sS 'https://bootstrap.pypa.io/get-pip.py' | python3.10
RUN python3.10 -m pip install --upgrade pip
RUN python3.10 -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
