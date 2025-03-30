FROM nvidia/cuda:11.8.0-base-ubuntu22.04
# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libgl1-mesa-glx libglib2.0-0 wget \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Install Python requirements first (for better caching)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
# Copy application code
COPY src/ /app/
# COPY app.py /app/
# COPY EmotionDetector.py /app/
# COPY SpotifyService.py /app/
# # Copy YOLO model
# COPY yolov8n-pose.pt /app/
# Health check endpoint for Cloud Run
EXPOSE 8000
# Use a startup script to handle initialization
RUN echo '#!/bin/bash\necho "Starting inference service..."\n\
    echo "CUDA devices: $NVIDIA_VISIBLE_DEVICES"\n\
    echo "NVIDIA capabilities: $NVIDIA_DRIVER_CAPABILITIES"\n\
    exec python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --log-level debug --ws websockets' > /app/start.sh && \
    chmod +x /app/start.sh
CMD ["/app/start.sh"]