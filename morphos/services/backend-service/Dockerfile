FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip cache purge
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire src directory structure
COPY src/ /app/

# Debug: List files
RUN ls -la /app/

# Use the exact command that worked before
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --ws websockets