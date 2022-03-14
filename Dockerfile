FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
                       build-essential \
                       ca-certificates \
                       wget \
                       curl \
                       unzip \
                       git \
                       ssh \
                       sudo \
                       vim \
                       default-jre \
                       python3.8-dev python3-pip python3-setuptools

RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN ln -s /usr/bin/python3.8 /usr/bin/python

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN python -m pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt
COPY ./scripts ./scripts
COPY ./configs ./configs
RUN sh scripts/build.sh
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install setuptools==59.5.0
RUN pip install promise==2.3

COPY . ./contrastive-learning-framework
