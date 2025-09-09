FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Arguments to build Docker Image using CUDA
ARG USE_CUDA=0
ARG TORCH_ARCH=

ENV AM_I_DOCKER True
ENV BUILD_WITH_CUDA "${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST "${TORCH_ARCH}"
ENV CUDA_HOME /usr/local/cuda-11.6/

RUN mkdir -p /home/appuser/Grounded-Segment-Anything
COPY . /home/appuser/Grounded-Segment-Anything/

RUN apt-get update && apt-get install --no-install-recommends -y \
    wget ffmpeg libsm6 libxext6 git nano vim build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /home/appuser/Grounded-Segment-Anything
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e segment_anything
RUN pip install --no-cache-dir --no-build-isolation -e GroundingDINO
