# Base image with Python and GPU support
FROM mcr.microsoft.com/devcontainers/python:1-3.10-bullseye

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    python3-dev \
    build-essential \
    cmake \
    zlib1g-dev \
    libjpeg-dev \
    xvfb \
    xorg \
    libsdl2-dev \
    libboost-all-dev \
    swig \
    ffmpeg \
    git \
    wget \
    parallel \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# Install GitHub CLI (gh)
RUN (type -p wget >/dev/null || (apt update && apt-get install wget -y)) \
    && mkdir -p -m 755 /etc/apt/keyrings \
    && wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt update \
    && apt install gh -y

# Copy your requirements.txt first (this is to cache dependencies installation)
COPY requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install --upgrade pip

RUN pip install uv

RUN uv pip install -r /tmp/requirements.txt --system

RUN pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126