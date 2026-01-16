FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Download model weights if not present
RUN if [ ! -f "./CRAFT-pytorch/weights/craft_mlt_25k.pth" ]; then \
    mkdir -p ./CRAFT-pytorch/weights && \
    python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ', './CRAFT-pytorch/weights/craft_mlt_25k.pth', quiet=False)"; \
    fi

# Create directories for images and results
RUN mkdir -p images result output

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/')" || exit 1

# Run API server
CMD ["python", "api.py"]
