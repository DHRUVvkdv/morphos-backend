from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import cv2
import logging
import os
import time
import glob
from typing import Dict, Any, List, Optional

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("morphos-inference")

# Create FastAPI app
app = FastAPI(
    title="Morphos Inference Service",
    description="YOLOv8-pose inference service with ONNX Runtime",
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


# Find ONNX model file
def find_model_file():
    # First check the expected path
    model_path = "/models/yolo/1/yolov8n-pose.onnx"
    if os.path.exists(model_path):
        logger.info(f"Found model at: {model_path}")
        return model_path

    # If not found, search more broadly
    model_files = glob.glob("/models/**/**.onnx", recursive=True)
    if model_files:
        logger.info(f"Found model at: {model_files[0]}")
        return model_files[0]

    logger.error("No ONNX model file found")
    return None


# Initialize ONNX session
MODEL_PATH = find_model_file()
ONNX_SESSION = None

try:
    import onnxruntime as ort

    if MODEL_PATH and os.path.exists(MODEL_PATH):
        # Log available providers
        logger.info(
            f"Available ONNX Runtime providers: {ort.get_available_providers()}"
        )

        # Try to use GPU if available, otherwise fall back to CPU
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 0  # Verbose logging

        try:
            logger.info(f"Initializing ONNX session with model: {MODEL_PATH}")
            ONNX_SESSION = ort.InferenceSession(
                MODEL_PATH, providers=providers, sess_options=session_options
            )

            # Log model info
            inputs = ONNX_SESSION.get_inputs()
            outputs = ONNX_SESSION.get_outputs()
            logger.info(f"Model inputs: {[i.name for i in inputs]}")
            logger.info(f"Model shapes: {[i.shape for i in inputs]}")
            logger.info(f"Model outputs: {[o.name for o in outputs]}")

            logger.info("ONNX Session initialized successfully with GPU support")
        except Exception as e:
            logger.error(f"Failed to initialize with GPU: {str(e)}")
            try:
                # Try with CPU only
                ONNX_SESSION = ort.InferenceSession(
                    MODEL_PATH, providers=["CPUExecutionProvider"]
                )
                logger.info("ONNX Session initialized with CPU provider only")
            except Exception as e2:
                logger.error(f"Failed to initialize with CPU: {str(e2)}")
                ONNX_SESSION = None
    else:
        logger.error(f"Model not found at {MODEL_PATH}")
        ONNX_SESSION = None
except Exception as e:
    logger.error(f"Error initializing ONNX Runtime: {str(e)}")
    ONNX_SESSION = None


# Define Pydantic models
class InferenceRequest(BaseModel):
    image: str  # Base64 encoded image


class Keypoint(BaseModel):
    x: float
    y: float
    confidence: float


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    class_name: str
    confidence: float
    keypoints: List[Keypoint] = []


class InferenceResponse(BaseModel):
    boxes: List[BoundingBox] = []
    processing_time: float = 0.0
    debug_info: Dict[str, Any] = {}


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


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image for YOLOv8-pose model"""
    # Resize to expected input size
    input_size = (640, 640)
    resized = cv2.resize(image, input_size)

    # Convert to RGB (YOLO expects RGB)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize pixel values
    normalized = rgb.astype(np.float32) / 255.0

    # Convert to expected input format
    input_data = np.transpose(normalized, (2, 0, 1))  # HWC to CHW
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

    return input_data


def run_onnx_inference(image: np.ndarray) -> np.ndarray:
    """Run inference with ONNX Runtime"""
    if ONNX_SESSION is None:
        logger.warning("ONNX session not initialized")
        return None

    try:
        # Preprocess image
        input_data = preprocess_image(image)

        # Get input name
        input_name = ONNX_SESSION.get_inputs()[0].name

        # Run inference
        logger.info(f"Running inference with input shape: {input_data.shape}")
        outputs = ONNX_SESSION.run(None, {input_name: input_data})

        # Log output info
        logger.info(f"Inference successful, output shape: {outputs[0].shape}")

        return outputs[0]
    except Exception as e:
        logger.error(f"Error during ONNX inference: {str(e)}")
        return None


def generate_fallback_detection(image: np.ndarray) -> List[BoundingBox]:
    """Generate mock detections as fallback"""
    logger.info("Using fallback detection mechanism")

    # Get image dimensions
    height, width = image.shape[:2]

    # For demo, return a person detection in center of frame
    boxes = []

    # Add a "person" detection in the middle
    center_x = width // 2
    center_y = height // 2
    box_width = width // 3
    box_height = height // 2

    # Generate some mock keypoints for a standing person
    keypoints = []
    # Head keypoint
    keypoints.append(Keypoint(x=center_x, y=center_y - box_height // 3, confidence=0.9))
    # Left shoulder
    keypoints.append(
        Keypoint(
            x=center_x - box_width // 4, y=center_y - box_height // 6, confidence=0.9
        )
    )
    # Right shoulder
    keypoints.append(
        Keypoint(
            x=center_x + box_width // 4, y=center_y - box_height // 6, confidence=0.9
        )
    )

    # Add more keypoints to reach 17 total (simplified)
    for i in range(14):
        keypoints.append(
            Keypoint(
                x=center_x + (((i % 3) - 1) * box_width / 5),
                y=center_y + ((i // 3) * box_height / 5),
                confidence=0.7,
            )
        )

    person_box = BoundingBox(
        x1=float(center_x - box_width // 2),
        y1=float(center_y - box_height // 2),
        x2=float(center_x + box_width // 2),
        y2=float(center_y + box_height // 2),
        class_id=0,
        class_name="person (fallback)",
        confidence=0.95,
        keypoints=keypoints[:17],  # Ensure we have exactly 17 keypoints
    )
    boxes.append(person_box)

    return boxes


def process_yolov8_pose_output(
    output: np.ndarray, original_image: np.ndarray
) -> List[BoundingBox]:
    """Process YOLOv8-pose output to get bounding boxes and keypoints"""
    if output is None:
        return []

    # Get original image dimensions
    orig_h, orig_w = original_image.shape[:2]

    try:
        # YOLOv8-pose output format is expected to be [batch, num_detections, 56]
        # Where 56 = 4 (bbox) + 1 (confidence) + 51 (17 keypoints × 3 values per keypoint)
        logger.info(f"Processing output with shape: {output.shape}")

        # Reshape if necessary
        if len(output.shape) == 3:
            detections = output  # Already in the expected format
        else:
            # Try to reshape based on expected format
            logger.info("Attempting to reshape output")
            detections = output.reshape(-1, 56)

        boxes = []
        for detection in detections:
            # First 4 values are x, y, w, h normalized
            x, y, w, h = detection[:4]

            # Convert to corner coordinates and scale to original image
            x1 = (x - w / 2) * orig_w
            y1 = (y - h / 2) * orig_h
            x2 = (x + w / 2) * orig_w
            y2 = (y + h / 2) * orig_h

            # Get detection confidence (index 4)
            confidence = detection[4]

            # Filter by confidence
            if confidence > 0.45:  # Lower threshold to catch more detections
                # Extract keypoints (indices 5 to 55 in groups of 3: x, y, conf)
                keypoints = []
                for i in range(17):  # 17 keypoints in COCO format
                    kp_idx = 5 + i * 3
                    kp_x = detection[kp_idx] * orig_w  # x coordinate
                    kp_y = detection[kp_idx + 1] * orig_h  # y coordinate
                    kp_conf = detection[kp_idx + 2]  # confidence

                    keypoints.append(
                        Keypoint(
                            x=float(kp_x), y=float(kp_y), confidence=float(kp_conf)
                        )
                    )

                # Create bounding box with keypoints
                boxes.append(
                    BoundingBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        class_id=0,  # YOLOv8-pose only detects people
                        class_name="person",
                        confidence=float(confidence),
                        keypoints=keypoints,
                    )
                )

        logger.info(f"Detected {len(boxes)} boxes in output")
        return boxes
    except Exception as e:
        logger.error(f"Error processing model output: {str(e)}")
        logger.error(f"Output shape: {output.shape if output is not None else 'None'}")
        return []


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}


# Debug endpoint
@app.get("/debug")
async def debug():
    # Find all ONNX models in the container
    model_files = glob.glob("/models/**/**.onnx", recursive=True)

    info = {
        "model_path": MODEL_PATH,
        "model_exists": MODEL_PATH is not None and os.path.exists(MODEL_PATH),
        "model_files_found": model_files,
        "models_dir_structure": {},
    }

    # Check models directory structure
    if os.path.exists("/models"):
        info["models_dir_exists"] = True
        for root, dirs, files in os.walk("/models"):
            rel_path = os.path.relpath(root, "/models")
            if rel_path == ".":
                rel_path = ""
            info["models_dir_structure"][rel_path] = files
    else:
        info["models_dir_exists"] = False

    # Check ONNX Runtime
    try:
        import onnxruntime as ort

        info["onnx_runtime_version"] = ort.__version__
        info["available_providers"] = ort.get_available_providers()
        info["onnx_session_initialized"] = ONNX_SESSION is not None

        if ONNX_SESSION is not None:
            # Get model details
            inputs = ONNX_SESSION.get_inputs()
            outputs = ONNX_SESSION.get_outputs()
            info["model_inputs"] = [{"name": i.name, "shape": i.shape} for i in inputs]
            info["model_outputs"] = [
                {"name": o.name, "shape": o.shape} for o in outputs
            ]
    except Exception as e:
        info["onnx_error"] = str(e)

    # Check GPU
    try:
        import subprocess

        gpu_info = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used",
                    "--format=csv,noheader",
                ]
            )
            .decode("utf-8")
            .strip()
        )
        info["gpu_info"] = gpu_info
    except Exception as e:
        info["gpu_error"] = str(e)

    return info


# Main inference endpoint
@app.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    start_time = time.time()
    debug_info = {"fallback_used": False}

    try:
        # Decode base64 image
        image = decode_base64_image(request.image)
        if image is None or image.size == 0:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Try to use ONNX model
        if ONNX_SESSION is not None:
            output = run_onnx_inference(image)
            if output is not None:
                boxes = process_yolov8_pose_output(output, image)
                debug_info["detection_count"] = len(boxes)

                # If no detections, use fallback
                if len(boxes) == 0:
                    logger.info("No detections from model, using fallback")
                    boxes = generate_fallback_detection(image)
                    debug_info["fallback_used"] = True
                    debug_info["fallback_reason"] = "No detections from model"
            else:
                logger.warning("ONNX inference failed, using fallback")
                boxes = generate_fallback_detection(image)
                debug_info["fallback_used"] = True
                debug_info["fallback_reason"] = "ONNX inference returned None"
        else:
            logger.warning("ONNX session not initialized, using fallback")
            boxes = generate_fallback_detection(image)
            debug_info["fallback_used"] = True
            debug_info["fallback_reason"] = "ONNX session not initialized"

        # Calculate processing time
        processing_time = time.time() - start_time

        return InferenceResponse(
            boxes=boxes, processing_time=processing_time, debug_info=debug_info
        )
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Inference processing failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    # Log startup information
    logger.info(f"Starting inference service")
    logger.info(f"Model path: {MODEL_PATH}")

    # Check models directory
    if os.path.exists("/models"):
        logger.info(f"Models directory exists")
        # List contents
        model_files = glob.glob("/models/**/**.onnx", recursive=True)
        logger.info(f"Found {len(model_files)} ONNX model files: {model_files}")
    else:
        logger.error(f"Models directory not found")

    # Run the FastAPI app
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="debug")
