from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import cv2
import logging
import requests
import json
import os
from typing import Dict, Any, List, Union, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("morphos-inference")

# Create FastAPI app
app = FastAPI(
    title="Morphos Inference Service",
    description="Dynamo (Triton) based inference service for object detection",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Triton server settings
TRITON_URL = os.environ.get("TRITON_URL", "http://localhost:8001")
MODEL_NAME = "yolov8n"


# Pydantic models
class InferenceRequest(BaseModel):
    image: str  # Base64 encoded image


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    class_name: str
    confidence: float


class InferenceResponse(BaseModel):
    boxes: List[BoundingBox] = []
    processing_time: float = 0.0


# Helper functions
def decode_base64_image(image_data: str) -> np.ndarray:
    """Decode base64 image to OpenCV format"""
    try:
        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        logger.error(f"Error decoding image: {str(e)}")
        return None


def preprocess_image(image: np.ndarray) -> Dict[str, Any]:
    """Preprocess image for YOLOv8 model"""
    # Resize to expected input size
    input_size = (640, 640)
    resized = cv2.resize(image, input_size)

    # Convert to RGB (YOLOv8 expects RGB)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize pixel values
    normalized = rgb.astype(np.float32) / 255.0

    # Convert to expected input format
    input_data = np.transpose(normalized, (2, 0, 1))  # HWC to CHW
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

    return {
        "inputs": [
            {
                "name": "images",
                "shape": list(input_data.shape),
                "datatype": "FP32",
                "data": input_data.flatten().tolist(),
            }
        ]
    }


def call_triton(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Send inference request to Triton server"""
    url = f"{TRITON_URL}/v2/models/{MODEL_NAME}/infer"

    try:
        response = requests.post(url, json=input_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to call Triton: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Inference request failed: {str(e)}"
        )


def postprocess_results(
    result: Dict[str, Any], original_image: np.ndarray
) -> List[BoundingBox]:
    """Process YOLOv8 output to get bounding boxes and classes"""
    # Define COCO classes for YOLOv8
    classes = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    # Get original image dimensions
    orig_h, orig_w = original_image.shape[:2]

    # Extract outputs
    outputs = result.get("outputs", [])[0]
    output_data = np.array(outputs["data"], dtype=np.float32).reshape((-1, 84))

    # Process detections
    boxes = []
    for detection in output_data:
        # First 4 values are x, y, w, h
        x, y, w, h = detection[:4]

        # Convert to corner coordinates and scale to original image
        x1 = (x - w / 2) * orig_w
        y1 = (y - h / 2) * orig_h
        x2 = (x + w / 2) * orig_w
        y2 = (y + h / 2) * orig_h

        # Get class probabilities (indexes 4-83)
        class_probs = detection[4:]
        class_id = np.argmax(class_probs)
        confidence = class_probs[class_id]

        # Filter by confidence
        if confidence > 0.5:  # Confidence threshold
            boxes.append(
                BoundingBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    class_id=int(class_id),
                    class_name=classes[class_id],
                    confidence=float(confidence),
                )
            )

    return boxes


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}


# Main inference endpoint
@app.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    try:
        import time

        start_time = time.time()

        # Decode base64 image
        image = decode_base64_image(request.image)
        if image is None or image.size == 0:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Preprocess for Triton
        input_data = preprocess_image(image)

        # Call Triton server
        triton_result = call_triton(input_data)

        # Process results
        boxes = postprocess_results(triton_result, image)

        # Calculate processing time
        processing_time = time.time() - start_time

        return InferenceResponse(boxes=boxes, processing_time=processing_time)

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Inference processing failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    # Get port from environment variable
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
