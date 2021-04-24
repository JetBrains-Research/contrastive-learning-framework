FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
                       build-essential \
                       ca-certificates \
                       wget \
                       unzip \
                       git \
                       ssh \
                       python3.8-dev python3.8-pip python3.8-setuptools

RUN ln -sf $(which python3) /usr/bin/python \
    && ln -sf $(which pip3) /usr/bin/pip

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /contrastive-learning-framework
COPY . /contrastive-learning-framework

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN sh scripts/build.sh
