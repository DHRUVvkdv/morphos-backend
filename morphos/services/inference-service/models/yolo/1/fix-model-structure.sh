#!/bin/bash
# This script ensures the Triton model repository is correctly structured

echo "Checking and fixing Triton model repository structure..."

# Check if model directory exists
MODEL_DIR="/models/yolo"
if [ ! -d "$MODEL_DIR" ]; then
  echo "Creating yolo model directory"
  mkdir -p "$MODEL_DIR/1"
fi

# Check if config.pbtxt exists
CONFIG_FILE="$MODEL_DIR/config.pbtxt"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Creating model config file"
  cat > "$CONFIG_FILE" << 'EOF'
name: "yolov8n-pose"
platform: "onnxruntime_onnx"
max_batch_size: 8
input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }
]
output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [ -1, 84 ]
  }
]
EOF
  echo "Created config.pbtxt with basic YOLO configuration"
fi

# Check if version directory exists
VERSION_DIR="$MODEL_DIR/1"
if [ ! -d "$VERSION_DIR" ]; then
  echo "Creating version directory"
  mkdir -p "$VERSION_DIR"
fi

# Check for model files in version directory
if [ -z "$(ls -A $VERSION_DIR)" ]; then
  echo "WARNING: No model files found in $VERSION_DIR"
  echo "Creating a placeholder file to prevent Triton errors"
  # For real use, you would need to download or copy in a real model file
  echo "This is a placeholder. Replace with actual ONNX model file." > "$VERSION_DIR/model.onnx.placeholder"
  echo "NOTE: You need to replace this placeholder with a real model file for inference to work"
fi

echo "Repository structure check complete."
echo "Current structure:"
find /models -type f | sort

echo "Config file content:"
cat "$CONFIG_FILE"