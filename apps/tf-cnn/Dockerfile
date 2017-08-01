# Snort in Docker
FROM gcr.io/tensorflow/tensorflow

ENV NPROCS 48

RUN apt-get update && \
    apt-get install -y \
    python-setuptools \
    python-pip \
    python-dev \
    wget \
    git \
    vim

ADD cifar10 /opt/cifar10
ADD cifar10_train /opt/cifar10_train

WORKDIR /opt/cifar10

