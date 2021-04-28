FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

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
                       default-jre \
                       python3.8-dev python3-pip python3-setuptools

RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN ln -s /usr/bin/python3.8 /usr/bin/python

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /contrastive-learning-framework
COPY . /contrastive-learning-framework

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN sh scripts/build.sh
