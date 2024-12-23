# -------------------------
# Base SageMaker image
# -------------------------

FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ENV PROJECT=calvin-sim

# https://gist.github.com/mohanpedala/1e2ff5661761d3abd0385e8223e16425
SHELL ["/bin/bash", "-ex", "-o", "pipefail", "-c"]


#=============================
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PYTORCH_VERSION=1.12.1+cu113
ENV TORCHVISION_VERSION=0.13.1+cu113
ENV TORCHAUDIO_VERSION=0.12.1

ENV CUDNN_VERSION=8.2.1.32-1+cuda11.3
ENV NCCL_VERSION=2.9.9-1+cuda11.3
ENV DEBIAN_FRONTEND=noninteractive

# RUN apt update 

# Install basic libraries
RUN apt-get update && apt-get install -y \
    build-essential cmake git curl docker.io vim wget ca-certificates

# Install python and pip
RUN apt-get install -y python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip

# Install pytorch
RUN pip install  \
    torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION} \
    --extra-index-url https://download.pytorch.org/whl/cu113

# TODO UNCOMMENT
# Install CUDNN and NCCL
RUN apt-get install -y \
    libcudnn8=${CUDNN_VERSION} \
    libnccl2=${NCCL_VERSION}

# Install extra packages (apt-get)
RUN apt-get install -y \
    ffmpeg \
    tmux

# UNCOMMENT
# Install extra packages (pip)
RUN pip install                 \
    tqdm==4.61.0                \
    boto3==1.17.83              \
    termcolor==1.1.0            \
    pyyaml==5.4.1               \
    wandb==0.10.31              \
    opencv-python==4.5.2.52     \
    flow_vis==0.1               \
    matplotlib==3.3.4           \
    fire==0.4.0                 \
    pyquaternion==0.9.9         \
    pandas==1.1.5               \
    xarray==0.16.2              \
    diskcache==5.2.1            \
    tenacity==7.0.0             \
    awscli==1.19.101            \
    ref==0.0.2.2                \
    positional-encodings==4.0.0 \
    einops==0.3.2               \
    wget==3.2                   \
    kornia==0.6.8               \
    lpips==0.1.4                \
    scikit-image==0.19.3        \
    timm==0.6.11

# Install CamViz dependencies
RUN pip install                 \
    pygame==2.0.1               \
    PyOpenGL==3.1.5             \
    PyOpenGL-accelerate==3.1.5
RUN apt-get install -y          \
    mesa-utils                  \
    freeglut3-dev               \
    libsdl2-2.0-0               \
    python-pygame

# Install CLIP
RUN pip install ftfy regex
RUN pip install git+https://github.com/openai/CLIP.git
# Install torch-scatter
RUN pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
# Install nuScenes devkit
RUN pip install nuscenes-devkit==1.1.7
# Upgrade protobuf
RUN pip install --upgrade protobuf==3.20.0
# Install transformer stuff
RUN pip install sacremoses huggingface_hub transformers
# Install open clip
RUN pip install open_clip_torch
# Install seaborn for extrinsic error map visualization
RUN pip install seaborn==0.12.2




#===========================



# Create an environment setup file separate from /etc/bash.bashrc, and source
# this file from /etc/bash.bashrc. The reason for this is that /etc/bash.bashrc
# doesn't execute when not running the shell interactively. If we have a separate
# file for environment setup, then we can source that file when executing
# non-interactively and still have the expected environment setup.
# This is useful for e.g., running tests with the expected environment.
# Takeaway: Instead of adding stuff to /etc/bash.bashrc, add it to this file!
ENV SHELL_SETUP_FILE=/usr/local/bin/${PROJECT}_env_setup.sh
RUN touch ${SHELL_SETUP_FILE} \
    && chmod +x ${SHELL_SETUP_FILE} \
    && echo "source ${SHELL_SETUP_FILE}" >> /etc/bash.bashrc

# -------------------------
# System tools
# -------------------------
# Configure apt for use in Docker.
RUN echo 'APT::Get::Assume-Yes "true";' | tee /etc/apt/apt.conf.d/90yes
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y \
    && apt-get install --no-install-recommends -q -y \
        locales \
        openssh-client openssh-server \
        lsb-release \
        cmake \
        curl \
        docker.io \
        ffmpeg \
        git \
        htop \
        libsm6 \
        libxext6 \
        libgl1  \
        libglib2.0-0 \
        libxrender-dev \
        libxext6 \
        libosmesa6-dev \
        libglfw3 \
        libjpeg-dev \
        libpng-dev \
        patchelf \
        python3-flake8-docstrings \
        python3-pip \
        python3-pyaudio \
        python3-pytest-cov \
        sudo \
        freeglut3-dev \
        software-properties-common \
        tmux \
        libasound2-dev \
    && rm -rf /var/lib/apt/lists/*
RUN locale-gen en_US.UTF-8
ENV \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# -------------------------
# Python packages
# -------------------------
RUN pip install --upgrade pip

RUN pip install wheel cmake==3.18.4



# Install basic libraries
RUN apt-get update && apt-get install -y \
    build-essential cmake git curl docker.io vim wget ca-certificates



# standard deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN pip install wheel cmake==3.18.4

# -------------------------
# Project code
# -------------------------
ENV PATH="/opt/ml/code:${PATH}"

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
ENV WORKSPACE=/opt/ml/code
COPY . ${WORKSPACE}




WORKDIR ${WORKSPACE}
ENV PYTHONPATH="${WORKSPACE}:$PYTHONPATH"
RUN git config --global --add safe.directory ${WORKSPACE}


# Add external packages to python path.
ENV EXTERNAL="${WORKSPACE}/external"


ENV PYTHONPATH="${EXTERNAL}/susie:$PYTHONPATH"
ENV PYTHONPATH="${EXTERNAL}/jaxrl_m:$PYTHONPATH"
ENV PYTHONPATH="${EXTERNAL}/LIBERO:$PYTHONPATH"

ENV PYTHONPATH="${WORKSPACE}/calvin_models:$PYTHONPATH"
ENV PYTHONPATH="${WORKSPACE}/calvin_env:$PYTHONPATH"
ENV PYTHONPATH="${WORKSPACE}/calvin_env/tacto:$PYTHONPATH"
ENV PYTHONPATH="${EXTERNAL}/urdfpy:$PYTHONPATH"
ENV PYTHONPATH="${EXTERNAL}/networkx:$PYTHONPATH"
ENV PYTHONPATH="${EXTERNAL}/pvdm:$PYTHONPATH"
ENV PYTHONPATH="${EXTERNAL}/lvdm:$PYTHONPATH"

RUN pip install --upgrade "jax[cuda11_pip]"==0.4.11 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    jaxlib==0.4.11 \
    flax==0.7.0 \
    optax==0.1.5 \
    distrax==0.1.3 \
    chex==0.1.7 


RUN pip install pyhash 
RUN pip install MulticoreTSNE

RUN alias p3=python3

RUN apt install -y libnvidia-gl-525
# RUN apt-get install -y libnvidia-gl-525-server
RUN apt install ffmpeg


RUN pip install --upgrade numpy 
RUN pip install pybullet==3.2.6

RUN pip install protobuf==3.20

# DynamiCrafter stuff
RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install open-clip-torch==2.12.0

# RUN python3 -u download_models.py 