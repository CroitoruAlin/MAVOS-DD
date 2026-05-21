# Copyright@SCLBD
# This Dockerfile aims to build the base image for Deepfakbench.
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

LABEL maintainer="Deepfake"

# Install dependencies outside of the base image
RUN DEBIAN_FRONTEND=noninteractive apt update && \
	apt install -y --no-install-recommends automake \
    build-essential  \
    ca-certificates  \
    libfreetype6-dev  \
    libtool  \
    pkg-config  \
    python-dev  \
    python-distutils-extra \
    python3.7-dev  \
    python3-pip \
    cmake \
	&& \
    rm -rf /var/lib/apt/lists/* \
    && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.7 0  \
    && \
    python3.7 -m pip install pip --upgrade 

WORKDIR /

RUN apt-get update -y && apt-get install -y git

# Install Python dependencies
RUN pip install --no-cache-dir certifi setuptools
RUN pip --no-cache-dir install dlib==19.24.0 \
    imageio==2.9.0 \
    imgaug==0.4.0 \
    scipy==1.7.3 \
    seaborn==0.11.2 \
    pyyaml==6.0 \
    imutils==0.5.4 \
    opencv-python==4.6.0.66 \
    scikit-image==0.19.2 \
    scikit-learn==1.0.2
    efficientnet-pytorch==0.7.1 \
    timm==0.6.12 \
    filterpy \
    simplejson \
    kornia \
    loralib
    segmentation-models-pytorch==0.3.2 \
    einops \
    tensorboard==2.10.1 \
    git+https://github.com/openai/CLIP.git

# RUN pip install --no-cache-dir \
torchtoolbox==0.1.8.2
pytorchvideo
transformers

ENV MODEL_NAME=deepfakebench

# Expose port
EXPOSE 6000
