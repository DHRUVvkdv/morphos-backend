FROM nvcr.io/nvidia/tritonserver:23.10-py3

# Install required packages
RUN pip install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn==0.27.1 \
    python-multipart==0.0.9 \
    numpy==1.24.4 \
    opencv-python-headless==4.8.1.78 \
    requests==2.31.0 \
    pydantic==2.5.2 \
    ultralytics==8.0.196

# Create directories
WORKDIR /app
RUN mkdir -p /models/yolov8n

# Download and convert YOLOv8 model
RUN python -c "from ultralytics import YOLO; \
    model = YOLO('yolov8n.pt'); \
    model.export(format='onnx', imgsz=640)"

# Move model to correct location
RUN mkdir -p /models/yolov8n/1 && \
    mv /app/yolov8n.onnx /models/yolov8n/1/model.onnx

# Create model configuration
RUN echo 'name: "yolov8n" \n\
    platform: "onnxruntime_onnx" \n\
    max_batch_size: 8 \n\
    input [{ \n\
    name: "images" \n\
    data_type: TYPE_FP32 \n\
    dims: [ 3, 640, 640 ] \n\
    }] \n\
    output [{ \n\
    name: "output0" \n\
    data_type: TYPE_FP32 \n\
    dims: [ -1, 84 ] \n\
    }]' > /models/yolov8n/config.pbtxt

# Copy service code
COPY src/ /app/

# Expose ports for FastAPI and Triton
EXPOSE 8000 8001 8002

# Start Triton in the background and then run the FastAPI service
CMD tritonserver --model-repository=/models --http-port=8001 --grpc-port=8002 & \
    uvicorn main:app --host 0.0.0.0 --port 8000