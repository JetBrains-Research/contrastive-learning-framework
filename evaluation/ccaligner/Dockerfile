FROM ubuntu:xenial

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing \
    && apt-get install -y \
        wget \
        bzip2 \
        ca-certificates \
        curl \
        grep \
        sed \
        dpkg \
        libglib2.0-0 \
        libxext6 \
        libsm6 \
        libxrender1 \
        git \
        libboost-dev \
        flex \
        build-essential


RUN git clone https://github.com/Tallic/ccaligner.git

COPY run_ccaligner.sh .
RUN chmod -R +x ccaligner run_ccaligner.sh

CMD [ "/bin/sh", "run_ccaligner.sh" ]