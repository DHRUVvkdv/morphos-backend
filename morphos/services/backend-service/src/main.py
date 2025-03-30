from fastapi import FastAPI, HTTPException, UploadFile, File, Request
import numpy as np
import cv2
import json
import requests
import logging
import base64
import os
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("morphos-inference")

# Create FastAPI app
app = FastAPI(
    title="Morphos Inference Service",
    description="Dynamo (Triton) based inference service for pose and emotion detection",
    version="0.1.0",
)

# Define inference models
TRITON_HOST = os.environ.get("TRITON_HOST", "localhost")
TRITON_HTTP_PORT = os.environ.get("TRITON_HTTP_PORT", "8001")
POSE_MODEL_NAME = "yolov8n-pose"
EMOTION_MODEL_NAME = "emotion-detection"  # Update with your actual emotion model name


# Pydantic models
class InferenceRequest(BaseModel):
    image: str  # Base64 encoded image


class KeyPoint(BaseModel):
    x: float
    y: float
    confidence: float


class PoseResult(BaseModel):
    keypoints: List[KeyPoint]
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float


class InferenceResponse(BaseModel):
    pose: Optional[PoseResult] = None
    emotion: Optional[str] = None
    rep_count: int = 0
    form_quality: str = "unknown"


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}


# Helper functions
def decode_base64_image(image_data: str) -> np.ndarray:
    """Decode base64 image to OpenCV format"""
    if "," in image_data:
        image_data = image_data.split(",")[1]

    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def preprocess_image_for_pose(image: np.ndarray) -> Dict[str, Any]:
    """Preprocess image for YOLOv8 pose model"""
    # Resize to expected input size
    resized = cv2.resize(image, (640, 640))

    # Normalize pixel values
    normalized = resized.astype(np.float32) / 255.0

    # Convert to expected input format
    input_data = np.transpose(normalized, (2, 0, 1))  # HWC to CHW
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

    return {
        "inputs": [
            {
                "name": "images",
                "shape": input_data.shape,
                "datatype": "FP32",
                "data": input_data.flatten().tolist(),
            }
        ]
    }


def preprocess_image_for_emotion(image: np.ndarray) -> Dict[str, Any]:
    """Preprocess image for emotion detection model"""
    # Implement based on your emotion model requirements
    # This is a placeholder - adjust based on your model's input requirements
    resized = cv2.resize(image, (224, 224))

    # Normalize pixel values
    normalized = resized.astype(np.float32) / 255.0

    # Convert to expected input format
    input_data = np.transpose(normalized, (2, 0, 1))  # HWC to CHW
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

    return {
        "inputs": [
            {
                "name": "input",
                "shape": input_data.shape,
                "datatype": "FP32",
                "data": input_data.flatten().tolist(),
            }
        ]
    }


def call_dynamo_http(model_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Send inference request to Dynamo (Triton) server via HTTP"""
    url = f"http://{TRITON_HOST}:{TRITON_HTTP_PORT}/v2/models/{model_name}/infer"

    try:
        response = requests.post(url, json=input_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to call Dynamo for model {model_name}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Inference request failed: {str(e)}"
        )


def process_pose_response(
    response: Dict[str, Any], original_image: np.ndarray
) -> PoseResult:
    """Process YOLOv8 pose model response"""
    # This processing depends on your specific model output format
    # Adjust according to your model's output structure
    outputs = response.get("outputs", [])

    # Extract keypoints, assuming first output is keypoints
    # This is placeholder logic - adjust based on actual model output
    keypoints_output = next((o for o in outputs if o["name"] == "output0"), None)
    if not keypoints_output:
        raise ValueError("No keypoints in model output")

    # Parse keypoints (17 keypoints for COCO format)
    # Format: [x1, y1, conf1, x2, y2, conf2, ...]
    raw_data = np.array(keypoints_output["data"], dtype=np.float32)

    # Assuming first detection if any
    if len(raw_data) > 0:
        # Extract first detection
        # Format varies by model, this is just an example
        detection = raw_data[:51].reshape(-1, 3)  # 17 keypoints x 3 values (x, y, conf)

        # Convert to response format
        keypoints = []
        for i in range(detection.shape[0]):
            keypoints.append(
                KeyPoint(
                    x=float(detection[i, 0]),
                    y=float(detection[i, 1]),
                    confidence=float(detection[i, 2]),
                )
            )

        # Extract bounding box (placeholder logic)
        bbox = [
            0,
            0,
            original_image.shape[1],
            original_image.shape[0],
        ]  # Full image as fallback
        confidence = float(np.mean([kp.confidence for kp in keypoints]))

        return PoseResult(keypoints=keypoints, bbox=bbox, confidence=confidence)

    return None


def process_emotion_response(response: Dict[str, Any]) -> str:
    """Process emotion detection model response"""
    # This processing depends on your specific model output format
    # Adjust according to your model's output structure
    outputs = response.get("outputs", [])

    # Extract emotion probabilities
    # This is placeholder logic - adjust based on actual model output
    emotion_output = next((o for o in outputs if o["name"] == "output"), None)
    if not emotion_output:
        return "unknown"

    # Parse emotions
    emotions = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]
    probs = np.array(emotion_output["data"], dtype=np.float32)

    # Return emotion with highest probability
    if len(probs) == len(emotions):
        return emotions[np.argmax(probs)]

    return "unknown"


# Main inference endpoint
@app.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    try:
        # Decode base64 image
        image = decode_base64_image(request.image)
        if image is None or image.size == 0:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Store original dimensions
        original_height, original_width = image.shape[:2]

        # Initialize response
        response = InferenceResponse()

        # Run pose detection
        try:
            pose_input = preprocess_image_for_pose(image)
            pose_result = call_dynamo_http(POSE_MODEL_NAME, pose_input)
            response.pose = process_pose_response(pose_result, image)
        except Exception as e:
            logger.error(f"Pose detection failed: {str(e)}")
            # Continue with other inferences

        # Run emotion detection
        try:
            emotion_input = preprocess_image_for_emotion(image)
            emotion_result = call_dynamo_http(EMOTION_MODEL_NAME, emotion_input)
            response.emotion = process_emotion_response(emotion_result)
        except Exception as e:
            logger.error(f"Emotion detection failed: {str(e)}")
            # Continue with other inferences

        # Simple form quality estimation (placeholder logic)
        # In a real implementation, this would analyze pose keypoints to determine form
        if response.pose and response.pose.confidence > 0.7:
            response.form_quality = "good"
        elif response.pose and response.pose.confidence > 0.5:
            response.form_quality = "average"
        else:
            response.form_quality = "poor"

        # Rep counting logic would go here
        # This would involve tracking keypoints over time

        return response

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Inference processing failed: {str(e)}"
        )


# Batch inference endpoint
@app.post("/batch_inference")
async def batch_inference(request: Request):
    """Process multiple frames at once (useful for video analysis)"""
    # Implementation will be similar to single inference but handle multiple images
    # This endpoint would be useful for processing multiple frames in a single request
    return {"status": "not_implemented"}


# Direct file upload endpoint as an alternative to base64
@app.post("/inference/upload", response_model=InferenceResponse)
async def upload_inference(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None or image.size == 0:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # Convert to base64 for our existing pipeline
    _, buffer = cv2.imencode(".jpg", image)
    base64_image = base64.b64encode(buffer).decode("utf-8")

    # Use the same inference logic
    request = InferenceRequest(image=base64_image)
    return await run_inference(request)


# In your main.py file, make sure you have code like this at the bottom:
if __name__ == "__main__":
    import uvicorn
    import os

    # Get port from environment variable
    port = int(os.environ.get("PORT", 8080))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
