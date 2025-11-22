#!/bin/bash
set -e

echo "=== Starting Cloud Environment Setup ==="

# 1. System Dependencies
echo "Updating system packages..."
apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    python3-pip \
    python3-venv \
    libsndfile1 \
    ffmpeg

# 2. Python Environment
echo "Setting up Python environment..."
# If not in a virtual environment, create one (optional but good practice)
# python3 -m venv venv
# source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# 3. Install Unsloth (Optimized for H100/Ampere+)
echo "Installing Unsloth..."
# Using the latest installation command for Unsloth with CUDA 12.1 support (common on H100 images)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# 4. Install vLLM (for fast generation in EM loop)
echo "Installing vLLM..."
pip install vllm

# 5. Install Other Python Dependencies
echo "Installing project dependencies..."
pip install \
    transformers>=4.40.0 \
    datasets>=2.19.0 \
    huggingface_hub \
    python-dotenv \
    wandb \
    torch>=2.2.0 \
    sentencepiece \
    protobuf \
    scipy

# 6. Verify Installation
echo "Verifying installations..."
python3 -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import unsloth; print('Unsloth imported successfully')"
python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

echo "=== Setup Complete ==="
echo "Please ensure your .env file has HF_TOKEN and WANDB_API_KEY set."

