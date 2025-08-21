# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
#FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    unzip \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Create directories for data and models
RUN mkdir -p /app/data/train /app/models /app/outputs /app/logs

# Copy scripts
COPY requirements.txt .
COPY train.py .
COPY training_script.sh .
COPY data/train/train /app/data/train/
COPY data/train.csv /app/data/
COPY hyperopt_results.json /app/outputs/

# Install the GPU version of PyTorch directly
RUN python3 -m pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy any configuration files
# COPY config/ ./config/ 2>/dev/null || true

# Set permissions
RUN chmod +x training_script.sh

ENTRYPOINT ["/bin/bash", "/app/training_script.sh"]