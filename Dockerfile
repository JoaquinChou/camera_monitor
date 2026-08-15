# 基于官方 nvidia/cuda 镜像，指定 CUDA 版本 12.1
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# 设置环境变量以确保安装过程中的一些参数一致
ENV DEBIAN_FRONTEND=noninteractive

# 更新并安装必要的包，包括 SSH 服务
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    curl \
    openssh-server \
    git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && mkdir /var/run/sshd

# 安装 Python 3.10 和 pip
RUN apt-get install -y python3.10 python3.10-dev python3.10-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && apt-get install -y sudo

# 设置 Python3.10 为默认版本
RUN update-alternatives --install /usr/bin/python3 python /usr/bin/python3.10 1
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# 安装其他必需的 Python 包
RUN pip install --upgrade pip setuptools wheel jupyterlab

# 安装 git 和 ffmpeg（git 已装，但保留原样）
RUN apt-get install -y git ffmpeg

# 安装 MPI 库
RUN apt-get install -y libmpich-dev libopenmpi-dev

# 清理不必要的缓存以减小镜像大小
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# 切换默认工作目录
WORKDIR /workspace

# 允许root用户登录
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    mkdir -p /tensorboard/logs/fit

# 软链接（可能不存在 conda，忽略错误）
RUN if [ -f /opt/conda/bin/jupyter ]; then ln -s /opt/conda/bin/jupyter /usr/bin/jupyter; fi
RUN if [ -f /opt/conda/bin/tensorboard ]; then ln -s /opt/conda/bin/tensorboard  /usr/bin/tensorboard; fi

# ============ 新增内容开始 ============
# 安装 openmpi-bin 和 git-lfs（libmpich-dev 和 libopenmpi-dev 已装，但这里重新安装确保存在）
RUN apt-get update && apt-get install -y \
    libmpich-dev \
    openmpi-bin \
    libopenmpi-dev \
    git-lfs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 设置 LD_LIBRARY_PATH（永久生效）
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu

# 安装 NVIDIA 相关 Python 包及 flash-attn
RUN pip install tensorrt_llm==0.11.0 -U --extra-index-url https://pypi.nvidia.com --no-cache-dir \
    && pip install nvidia-ammo==0.9.5 \
    && pip install onnxruntime-gpu==1.18.0 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
    && pip install flash-attn --no-cache-dir --no-build-isolation
# ============ 新增内容结束 ============

# Expose Jupyter port & cmd
CMD /bin/bash -c "if [ -z \"${aidc_ssh_password}\" ]; then echo 'ERROR: aidc_ssh_password must be set!' && exit 1; fi && echo 'root:${aidc_ssh_password}' | chpasswd && /usr/sbin/sshd -D"
