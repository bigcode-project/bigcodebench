FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

# Setup Environment Variables
ENV CUDA_HOME=/usr/local/cuda \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"

# Setup System Utilities
RUN apt-get update --yes --quiet \
    && apt-get upgrade --yes --quiet \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
        apt-utils \
        autoconf \
        automake \
        bc \
        build-essential \
        ca-certificates \
        check \
        cmake \
        curl \
        dmidecode \
        emacs \
        g++\
        gcc \
        git \
        iproute2 \
        jq \
        kmod \
        libaio-dev \
        libcurl4-openssl-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
        libibverbs-dev \
        libnuma-dev \
        libnuma1 \
        libomp-dev \
        libsm6 \
        libssl-dev \
        libsubunit-dev \
        libsubunit0 \
        libtool \
        libxext6 \
        libxrender-dev \
        make \
        moreutils \
        net-tools \
        ninja-build \
        openssh-client \
        openssh-server \
        openssl \
        pkg-config \
        python3-dev \
        software-properties-common \
        sudo \
        unzip \
        util-linux \
        vim \
        wget \
        zlib1g-dev \
    && apt-get autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/

# Setup base Python to bootstrap Mamba
RUN add-apt-repository --yes ppa:deadsnakes/ppa \
    && apt-get update --yes --quiet
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        python3.11-lib2to3 \
        python3.11-gdbm \
        python3.11-tk \
        pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 999 \
    && update-alternatives --config python3 \
    && ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip

# Setup optimized Mamba environment with required PyTorch dependencies
RUN wget -O /tmp/Miniforge.sh https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Mambaforge-24.3.0-0-Linux-x86_64.sh \
    && bash /tmp/Miniforge.sh -b -p /Miniforge \
    && source /Miniforge/etc/profile.d/conda.sh \
    && source /Miniforge/etc/profile.d/mamba.sh \
    && mamba update -y -q -n base -c defaults mamba \
    && mamba create -y -q -n Code-Eval python=3.11 setuptools=69.5.1 \
    && mamba activate Code-Eval \
    && mamba install -y -q -c conda-forge \
        charset-normalizer \
        gputil \
        ipython \
        numpy \
        pandas \
        scikit-learn \
        wandb \
    && mamba install -y -q -c intel \
        "mkl==2023" \
        "mkl-static==2023" \
        "mkl-include==2023" \
    && mamba install -y -q -c pytorch magma-cuda121 \
    && mamba clean -a -f -y

# Install VLLM precompiled with appropriate CUDA and ensure PyTorch is installed form the same version channel
RUN source /Miniforge/etc/profile.d/conda.sh \
    && source /Miniforge/etc/profile.d/mamba.sh \
    && mamba activate Code-Eval \
    && pip install https://github.com/vllm-project/vllm/releases/download/v0.4.0/vllm-0.4.0-cp311-cp311-manylinux1_x86_64.whl \
        --extra-index-url https://download.pytorch.org/whl/cu121

# Install Flash Attention
RUN source /Miniforge/etc/profile.d/conda.sh \
    && source /Miniforge/etc/profile.d/mamba.sh \
    && mamba activate Code-Eval \
    && export MAX_JOBS=$(($(nproc) - 2)) \
    && pip install --no-cache-dir ninja packaging psutil \
    && pip install flash-attn==2.5.8 --no-build-isolation

# Acquire benchmark code to local
RUN git clone https://github.com/bigcode-project/code-eval.git /bigcodebench

# Install Code-Eval and pre-load the dataset
RUN source /Miniforge/etc/profile.d/conda.sh \
    && source /Miniforge/etc/profile.d/mamba.sh \
    && mamba activate Code-Eval \
    && pip install bigcodebench --upgrade \
    && python -c "from bigcodebench.data import get_bigcodebench; get_bigcodebench()"

WORKDIR /bigcodebench

# Declare an argument for the huggingface token
ARG HF_TOKEN
RUN if [[ -n "$HF_TOKEN" ]] ; then /Miniforge/envs/Code-Eval/bin/huggingface-cli login --token $HF_TOKEN ; \
    else echo "No HuggingFace token specified. Access to gated or private models will be unavailable." ; fi

ENTRYPOINT ["/Miniforge/envs/Code-Eval/bin/python", "-m", "bigcodebench.generate"]