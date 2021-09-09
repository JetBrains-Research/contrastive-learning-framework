FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu18.04

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
                    vim

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda init bash

RUN git clone https://github.com/maximzubkov/CodeGen.git
RUN cd CodeGen && bash install_env.sh
