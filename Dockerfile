# Use lightweight Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for pygame & rendering)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libsmpeg-dev \
    libportmidi-dev \
    libavformat-dev \
    libswscale-dev \
    libjpeg-dev \
    libfreetype6-dev \
    libx11-6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Default command
CMD ["python", "train.py"]