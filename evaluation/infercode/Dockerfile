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
                       zip \
                       default-jre \
                       python3.8-dev python3-pip python3-setuptools

RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN ln -s /usr/bin/python3.8 /usr/bin/python

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN python -m pip install --upgrade pip
RUN pip install infercode==0.0.28

COPY run_infercode.py .
CMD [ "python", "run_infercode.py" ]
