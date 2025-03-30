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
import logging

logger = logging.getLogger("morphos-inference")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def log_gpu_info():
    try:
        import subprocess

        gpu_info = subprocess.check_output(
            ["nvidia-smi"], stderr=subprocess.STDOUT
        ).decode("utf-8")
        logger.info(f"GPU INFO: {gpu_info}")
    except Exception as e:
        logger.error(f"Error getting GPU info: {str(e)}")


log_gpu_info()

is_cloud_run = os.environ.get("K_SERVICE") is not None
is_gpu_enabled = os.environ.get("NVIDIA_VISIBLE_DEVICES") is not None

if is_gpu_enabled:
    logger.info(f"GPU environment detected: {os.environ.get('NVIDIA_VISIBLE_DEVICES')}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    logger.info("No GPU environment detected, using CPU")


# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("morphos-inference")


def check_model_availability():
    logger.info("==== Model Availability Check ====")
    model_path = "/models/yolo/1/yolov8n-pose.onnx"
    if os.path.exists(model_path):
        logger.info(f"✅ Model file exists at: {model_path}")
        logger.info(f"✅ Model file size: {os.path.getsize(model_path)} bytes")
    else:
        logger.error(f"❌ Model file DOES NOT EXIST at: {model_path}")
        # Check what's in the models directory
        if os.path.exists("/models"):
            logger.info("✅ /models directory exists")
            # List all files in the models directory recursively
            found_files = []
            for root, dirs, files in os.walk("/models"):
                for file in files:
                    found_files.append(os.path.join(root, file))
            logger.info(f"Files in /models directory: {found_files}")
        else:
            logger.error("❌ /models directory DOES NOT EXIST")


# Run the check early
check_model_availability()

logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Files in current directory: {os.listdir('.')}")
try:
    logger.info(f"Files in models directory: {os.listdir('models')}")
    logger.info(f"Files in models/yolo directory: {os.listdir('models/yolo')}")
    logger.info(f"Files in models/yolo/1 directory: {os.listdir('models/yolo/1')}")
except Exception as e:
    logger.error(f"Error listing directory structure: {e}")

# Check ONNX Runtime installation
try:
    import onnxruntime as ort

    logger.info(f"✅ ONNX Runtime version: {ort.__version__}")
    logger.info(f"✅ Available providers: {ort.get_available_providers()}")
except ImportError as e:
    logger.error(f"❌ Failed to import onnxruntime: {str(e)}")
except Exception as e:
    logger.error(f"❌ Error with onnxruntime: {str(e)}")

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
    """Find ONNX model file"""
    # List of possible paths to try
    model_paths = [
        "/models/yolo/1/yolov8n-pose.onnx",  # Standard path in Docker
        "/models/yolov8n-pose.onnx",  # Simpler path
        "models/yolo/1/yolov8n-pose.onnx",  # Relative path
    ]

    # Try each path
    for path in model_paths:
        if os.path.exists(path):
            file_size = os.path.getsize(path)
            logger.info(f"Found model at: {path} (size: {file_size} bytes)")
            return path

    # If not found, search more broadly
    search_paths = ["/models", "models", "."]
    for base_path in search_paths:
        if not os.path.exists(base_path):
            continue

        for root, _, files in os.walk(base_path):
            for file in files:
                if file.endswith(".onnx"):
                    path = os.path.join(root, file)
                    logger.info(f"Found model at: {path}")
                    return path

    logger.error("No ONNX model file found")
    return None


# Initialize ONNX session
def initialize_onnx_session():
    global ONNX_SESSION
    global MODEL_PATH

    MODEL_PATH = find_model_file()
    if not MODEL_PATH:
        logger.error("No model file found")
        return False

    try:
        import onnxruntime as ort

        available_providers = ort.get_available_providers()
        logger.info(f"Available ONNX Runtime providers: {available_providers}")

        # Start with CPU provider for reliability
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        try:
            logger.info(f"Initializing ONNX session with CPU provider")
            ONNX_SESSION = ort.InferenceSession(
                MODEL_PATH,
                providers=["CPUExecutionProvider"],
                sess_options=session_options,
            )
            logger.info("ONNX Session initialized with CPU")

            # Get model info
            input_name = ONNX_SESSION.get_inputs()[0].name
            input_shape = ONNX_SESSION.get_inputs()[0].shape
            output_name = ONNX_SESSION.get_outputs()[0].name
            logger.info(f"Model input: {input_name} with shape {input_shape}")
            logger.info(f"Model output: {output_name}")

            # Try GPU if available
            if "CUDAExecutionProvider" in available_providers:
                try:
                    logger.info("Attempting to use GPU acceleration")
                    gpu_session = ort.InferenceSession(
                        MODEL_PATH,
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                        sess_options=session_options,
                    )

                    # Test the GPU session with a simple inference
                    import numpy as np

                    test_input = np.zeros(input_shape, dtype=np.float32)
                    _ = gpu_session.run(None, {input_name: test_input})

                    # If successful, use the GPU session
                    ONNX_SESSION = gpu_session
                    logger.info("Successfully switched to GPU-accelerated session")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to initialize GPU: {str(e)}")
                    logger.info("Using CPU session instead")
                    return True
            else:
                logger.info("No GPU provider available, using CPU session")
                return True

        except Exception as e:
            logger.error(f"Failed to initialize ONNX session: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    except ImportError as e:
        logger.error(f"Error importing onnxruntime: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return False


# Initialize ONNX at startup
ONNX_SESSION = None
initialize_onnx_session()


@app.get("/onnx-test")
async def onnx_test():
    """Test ONNX model loading and run a test inference"""
    results = {
        "timestamp": time.time(),
        "model_path": MODEL_PATH,
        "model_exists": MODEL_PATH is not None and os.path.exists(MODEL_PATH),
        "onnx_session_initialized": ONNX_SESSION is not None,
        "tests": [],
    }

    # Test 1: Check model file
    test1 = {"name": "Model file check"}
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        test1["status"] = "PASS"
        test1["file_size"] = os.path.getsize(MODEL_PATH)
    else:
        test1["status"] = "FAIL"
        if os.path.exists("/models"):
            files = []
            for root, _, filenames in os.walk("/models"):
                for filename in filenames:
                    files.append(os.path.join(root, filename))
            test1["available_files"] = files
        else:
            test1["error"] = "/models directory does not exist"
    results["tests"].append(test1)

    # Test 2: Check ONNX Runtime
    test2 = {"name": "ONNX Runtime check"}
    try:
        import onnxruntime as ort

        test2["status"] = "PASS"
        test2["version"] = ort.__version__
        test2["providers"] = ort.get_available_providers()
    except Exception as e:
        test2["status"] = "FAIL"
        test2["error"] = str(e)
    results["tests"].append(test2)

    # Test 3: Test model loading
    test3 = {"name": "Model loading test"}
    if ONNX_SESSION is not None:
        test3["status"] = "PASS"
        try:
            inputs = ONNX_SESSION.get_inputs()
            outputs = ONNX_SESSION.get_outputs()
            test3["input_info"] = [
                {"name": i.name, "shape": list(i.shape)} for i in inputs
            ]
            test3["output_info"] = [
                {
                    "name": o.name,
                    "shape": list(o.shape) if hasattr(o, "shape") else None,
                }
                for o in outputs
            ]
        except Exception as e:
            test3["error_getting_model_info"] = str(e)
    else:
        test3["status"] = "FAIL"
        test3["error"] = "ONNX_SESSION is None"
    results["tests"].append(test3)

    # Test 4: Simple inference test
    test4 = {"name": "Inference test"}
    if ONNX_SESSION is not None:
        try:
            # Create a dummy input
            input_name = ONNX_SESSION.get_inputs()[0].name
            input_shape = ONNX_SESSION.get_inputs()[0].shape

            import numpy as np

            dummy_input = np.zeros(input_shape, dtype=np.float32)

            # Run inference
            start = time.time()
            outputs = ONNX_SESSION.run(None, {input_name: dummy_input})
            end = time.time()

            test4["status"] = "PASS"
            test4["inference_time"] = end - start
            test4["output_shapes"] = [list(o.shape) for o in outputs]
        except Exception as e:
            test4["status"] = "FAIL"
            test4["error"] = str(e)
            import traceback

            test4["traceback"] = traceback.format_exc()
    else:
        test4["status"] = "SKIP"
        test4["reason"] = "ONNX_SESSION is None"
    results["tests"].append(test4)

    return results


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
        # Handle data URL format (e.g., "data:image/jpeg;base64,...")
        if "," in image_data:
            image_data = image_data.split(",")[1]

        # Decode base64
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)

        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            logger.error("Decoded image is None")
            return None

        logger.info(f"Successfully decoded image with shape: {image.shape}")

        return image
    except Exception as e:
        logger.error(f"Error decoding image: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return None


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image for YOLOv8-pose model"""
    # Get original image dimensions for aspect ratio preservation
    original_height, original_width = image.shape[:2]

    # Calculate scale to maintain aspect ratio
    input_size = (640, 640)
    ratio = min(input_size[0] / original_width, input_size[1] / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # Resize with preserved aspect ratio
    resized = cv2.resize(image, (new_width, new_height))

    # Create a black canvas of target size
    canvas = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)

    # Paste the resized image centered on the canvas
    offset_x = (input_size[0] - new_width) // 2
    offset_y = (input_size[1] - new_height) // 2
    canvas[offset_y : offset_y + new_height, offset_x : offset_x + new_width] = resized

    # Convert to RGB (YOLO expects RGB)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

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
        # Log original image details
        logger.info(f"Original image shape: {image.shape}")

        # Preprocess image
        input_data = preprocess_image(image)
        logger.info(f"Preprocessed input shape: {input_data.shape}")

        # Get input name
        input_name = ONNX_SESSION.get_inputs()[0].name
        logger.info(f"Model input name: {input_name}")

        # Run inference
        logger.info(f"Running inference with input shape: {input_data.shape}")
        outputs = ONNX_SESSION.run(None, {input_name: input_data})

        # Log output details
        for i, output in enumerate(outputs):
            logger.info(f"Output {i} shape: {output.shape}")
            logger.info(f"Output {i} min/max values: {output.min()}/{output.max()}")

        # Return first output
        return outputs[0]
    except Exception as e:
        logger.error(f"Error during ONNX inference: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
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
        # Log the output shape to understand what we're working with
        logger.info(f"Processing output with shape: {output.shape}")

        # Confidence threshold - lowered for better detection
        conf_thres = 0.25  # Reduced from 0.45

        # YOLOv8-pose output format should be [batch, 56, num_anchors]
        # Parse based on the actual shape we get
        boxes = []

        if len(output.shape) == 3 and output.shape[1] == 56:
            # This is the expected output format for YOLOv8-pose
            # Reshape to [num_anchors, 56]
            output = output[0].transpose(1, 0)  # Now [num_anchors, 56]

            # Get detections with confidence above threshold
            mask = output[4] > conf_thres
            detections = output[:, mask]

            if detections.size > 0:
                for i in range(detections.shape[1]):
                    # Get bounding box coordinates (x, y, w, h format)
                    x, y, w, h = detections[0:4, i]

                    # Convert to corner coordinates
                    x1 = (x - w / 2) * orig_w
                    y1 = (y - h / 2) * orig_h
                    x2 = (x + w / 2) * orig_w
                    y2 = (y + h / 2) * orig_h

                    # Get confidence
                    confidence = detections[4, i]

                    # Parse keypoints (17 keypoints, each with x, y, confidence)
                    keypoints = []
                    for j in range(17):
                        kp_idx = 5 + j * 3
                        kp_x = detections[kp_idx, i] * orig_w
                        kp_y = detections[kp_idx + 1, i] * orig_h
                        kp_conf = detections[kp_idx + 2, i]

                        keypoints.append(
                            Keypoint(
                                x=float(kp_x), y=float(kp_y), confidence=float(kp_conf)
                            )
                        )

                    # Add to boxes
                    boxes.append(
                        BoundingBox(
                            x1=float(x1),
                            y1=float(y1),
                            x2=float(x2),
                            y2=float(y2),
                            class_id=0,
                            class_name="person",
                            confidence=float(confidence),
                            keypoints=keypoints,
                        )
                    )
        else:
            # Try alternate format - reshape based on what we know
            logger.info(
                f"Output shape doesn't match expected format. Attempting to reshape."
            )

            # Check if output matches [1, 56, 8400] shape from debug endpoint
            if output.shape == (1, 56, 8400):
                # Transpose to [1, 8400, 56]
                output = np.transpose(output, (0, 2, 1))
                # Get first element [8400, 56]
                output = output[0]

                # Filter by confidence
                confident_detections = output[output[:, 4] > conf_thres]

                for detection in confident_detections:
                    # First 4 values are x, y, w, h normalized
                    x, y, w, h = detection[:4]

                    # Convert to corner coordinates and scale to original image
                    x1 = (x - w / 2) * orig_w
                    y1 = (y - h / 2) * orig_h
                    x2 = (x + w / 2) * orig_w
                    y2 = (y + h / 2) * orig_h

                    # Get detection confidence (index 4)
                    confidence = detection[4]

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
        import traceback

        logger.error(traceback.format_exc())
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
