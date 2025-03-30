FROM nvcr.io/nvidia/tritonserver:23.10-py3

# Install required packages
RUN pip3 install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn==0.27.1 \
    python-multipart==0.0.9 \
    numpy==1.24.4 \
    opencv-python-headless==4.8.1.78 \
    requests==2.31.0 \
    pydantic==2.5.2

# Create directories
WORKDIR /app

# Copy service code
COPY src/ /app/

# Copy models
COPY models/ /models/

# Expose ports for FastAPI and Triton
EXPOSE 8000 8001 8002

# Start Triton in the background and then run the FastAPI service
CMD tritonserver --model-repository=/models --http-port=8001 --grpc-port=8002 & \
    uvicorn main:app --host 0.0.0.0 --port 8000