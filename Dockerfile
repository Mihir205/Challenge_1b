# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PyMuPDF
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (to cache pip installs)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the pre-downloaded model (offline mode)
COPY models ./models

# Copy the rest of the project
COPY . .

# Set environment variable for HuggingFace offline mode
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

# Default command to run the app
CMD ["python", "main.py"]
