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

ADD TensorFlow-Examples /opt/TensorFlow-Examples

WORKDIR /opt/TensorFlow-Examples/examples/3_NeuralNetworks

