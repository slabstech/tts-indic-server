
# Use NVIDIA CUDA 12.8.0 with cuDNN development tools on Ubuntu 22.04
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04


# Set working directory
WORKDIR /app


# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    sudo \
    wget \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip setuptools

RUN pip install --upgrade pip setuptools wheel


RUN pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

#RUN pip install flash-attn --no-build-isolation


# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of the application code
COPY . .


# Create a non-root user and set ownership
RUN useradd -ms /bin/bash appuser \
    && chown -R appuser:appuser /app


# Switch to non-root user
USER appuser


# Expose port for the API
EXPOSE 7860


# Run the TTS API server
CMD ["python", "/app/src/gh200/main.py", "--host", "0.0.0.0", "--port", "7864", "--config", "config_two"]