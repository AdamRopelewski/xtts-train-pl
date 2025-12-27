FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

# Install system packages and Python
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       sudo ca-certificates gnupg2 lsb-release curl wget git build-essential \
       python3 python3-pip python3-venv python-is-python3 \
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswresample-dev \
        libswscale-dev \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*



# Fix faster-whisper looking for libcublas.so.12
RUN ln -s /usr/local/cuda/lib64/libcublas.so.13 /usr/local/cuda/lib64/libcublas.so.12 \
 && ln -s /usr/local/cuda/lib64/libcublasLt.so.13 /usr/local/cuda/lib64/libcublasLt.so.12


# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
