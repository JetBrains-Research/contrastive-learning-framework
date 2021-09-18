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
WORKDIR /CodeGen

RUN conda create --name env
RUN conda config --add channels conda-forge
RUN conda config --add channels pytorch

RUN conda install -n env -y pytorch torchvision torchaudio cudatoolkit=11.0 six scikit-learn stringcase transformers ply slimit astunparse submitit
RUN /root/miniconda3/envs/env/bin/pip install cython

WORKDIR /CodeGen/codegen_sources/model/tools
RUN git clone https://github.com/glample/fastBPE.git

WORKDIR /CodeGen/codegen_sources/model/tools/fastBPE
RUN g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
RUN /root/miniconda3/envs/env/bin/python setup.py install

WORKDIR /CodeGen

RUN mkdir tree-sitter
WORKDIR /CodeGen/tree-sitter
RUN git clone https://github.com/tree-sitter/tree-sitter-cpp.git
RUN git clone https://github.com/tree-sitter/tree-sitter-java.git
RUN git clone https://github.com/tree-sitter/tree-sitter-python.git

WORKDIR /CodeGen
RUN /root/miniconda3/envs/env/bin/pip install -r ci_requirements.txt

WORKDIR /CodeGen

RUN wget https://dl.fbaipublicfiles.com/transcoder/pre_trained_models/TransCoder_model_1.pth
RUN wget https://dl.fbaipublicfiles.com/transcoder/pre_trained_models/TransCoder_model_2.pth

RUN /root/miniconda3/envs/env/bin/pip install sacrebleu=="1.2.11" javalang tree_sitter psutil fastBPE
RUN /root/miniconda3/envs/env/bin/pip install hydra-core --upgrade --pre
RUN /root/miniconda3/envs/env/bin/pip install black==19.10b0

COPY run_transcoder.py .

CMD [ "python", "run_transcoder.py", "TransCoder_model_1.pth" ]
CMD [ "python", "run_transcoder.py", "TransCoder_model_2.pth" ]