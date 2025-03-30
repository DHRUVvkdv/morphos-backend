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
COPY models/ /models/

# Verify model exists
RUN echo "Checking model files:" && \
    ls -la /models/yolo/1/ && \
    echo "Model size: $(du -sh /models/yolo/1/yolov8n-pose.onnx)"

# Health check for faster startup recognition
HEALTHCHECK --interval=3s --timeout=3s --start-period=30s --retries=3 \
    CMD wget -q -O - http://localhost:8000/health || exit 1

# Set environment variables for CUDA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Expose port
EXPOSE 8000

# Use a startup script to handle initialization
RUN echo '#!/bin/bash\necho "Starting inference service..."\n\
    echo "CUDA devices: $NVIDIA_VISIBLE_DEVICES"\n\
    echo "NVIDIA capabilities: $NVIDIA_DRIVER_CAPABILITIES"\n\
    exec python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level debug' > /app/start.sh && \
    chmod +x /app/start.sh

CMD ["/app/start.sh"]