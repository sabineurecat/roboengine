# Use NVIDIA CUDA 11.8 base image with cuDNN support and Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

LABEL maintainer="your-email@example.com"

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python, and symlink python -> python3
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-opencv \
    build-essential \
    cmake \
    libomp-dev \
    libffi-dev \
    libdbus-1-dev \
    libgirepository1.0-dev \
    gir1.2-glib-2.0 \
    pkg-config \
    ninja-build \
    libcairo2-dev \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Install Meson (specific version)
RUN pip3 install --no-cache-dir meson==0.63.3

# Install PyTorch with CUDA 11.8 support
RUN pip3 install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio --index-url https://download.pytorch.org/whl/cu118

# Clone and install SAM2
RUN git clone https://github.com/facebookresearch/sam2.git /workspace/sam2 && \
    cd /workspace/sam2 && \
    git submodule update --init --recursive && \
    pip3 install -e ".[notebooks]"

# Copy project files
COPY . /workspace
WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip3 install --no-cache-dir -r /workspace/requirements.txt && \
    rm -rf /root/.cache/pip

# Install specific version of spaCy model
RUN wget https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl && \
    pip3 install --no-cache-dir en_core_web_sm-3.8.0-py3-none-any.whl && \
    rm en_core_web_sm-3.8.0-py3-none-any.whl

# Set Python path for imports
ENV PYTHONPATH=/workspace/robo_engine:$PYTHONPATH

# Default command
CMD ["bash"]
