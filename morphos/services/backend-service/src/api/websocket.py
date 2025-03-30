from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    HTTPException,
    Request,
)
import logging
import json
import base64
import cv2
import numpy as np
import asyncio
import aiohttp
import os
import time
from typing import Dict, Any, List, Optional

from core.managers import ConnectionManager
from core.security import verify_token
from config.settings import Settings

logger = logging.getLogger("morphos-websocket")
websocket_router = APIRouter()

# Initialize the connection manager
manager = ConnectionManager()

# Load settings
settings = Settings()

# Initialize session storage for tracking client state
session_data = {}


# Add this helper function for CORS
async def cors_validation(websocket: WebSocket):
    """Validate CORS request and handle preflight."""
    request = websocket.scope.get("request", {"headers": {}})
    origin = next(
        (
            value.decode()
            for name, value in websocket.scope.get("headers", [])
            if name.decode().lower() == "origin"
        ),
        None,
    )

    logger.info(f"WebSocket connection attempt from origin: {origin}")

    # Always accept connections in development mode
    if settings.DEBUG:
        logger.info("DEBUG mode - accepting all WebSocket connections")
        return True

    # Check if origin is allowed
    if origin and (origin in settings.CORS_ORIGINS or "*" in settings.CORS_ORIGINS):
        logger.info(f"Origin {origin} is allowed")
        return True

    logger.warning(
        f"Origin {origin} is not allowed. CORS_ORIGINS: {settings.CORS_ORIGINS}"
    )
    return False


async def call_inference_service(frame_data: str) -> Dict[str, Any]:
    """
    Call the inference service with a video frame

    Args:
        frame_data: Base64 encoded image data

    Returns:
        Dict containing analysis results from the inference service
    """
    inference_url = os.environ.get(
        "INFERENCE_SERVICE_URL", "http://inference-service:8000"
    )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{inference_url}/inference",
                json={"image": frame_data},
                timeout=aiohttp.ClientTimeout(total=5),  # 5s timeout
            ) as response:
                if response.status != 200:
                    logger.error(f"Inference service error: {response.status}")
                    return {"error": f"Inference service error: {response.status}"}

                return await response.json()
    except asyncio.TimeoutError:
        logger.error("Inference service timeout")
        return {"error": "Inference service timeout"}
    except Exception as e:
        logger.error(f"Error calling inference service: {str(e)}")
        return {"error": f"Error calling inference service: {str(e)}"}


class RepCounter:
    """Simple rep counter based on keypoint movement"""

    def __init__(self, keypoint_idx=9):  # Index 9 is typically mid-hip in COCO format
        self.keypoint_idx = keypoint_idx
        self.positions = []
        self.max_positions = 10  # Store last N positions
        self.going_up = None
        self.rep_count = 0
        self.last_rep_time = 0

    def update(self, keypoints: List[Dict[str, Any]]) -> int:
        """Update with new keypoints and determine if a rep was completed"""
        if not keypoints or len(keypoints) <= self.keypoint_idx:
            return self.rep_count

        # Get current Y position of tracked keypoint
        current_y = keypoints[self.keypoint_idx].get("y", 0)
        current_confidence = keypoints[self.keypoint_idx].get("confidence", 0)

        # If confidence is too low, skip this update
        if current_confidence < 0.5:
            return self.rep_count

        # Add to position history
        self.positions.append(current_y)
        if len(self.positions) > self.max_positions:
            self.positions.pop(0)

        # Need at least a few positions to detect movement
        if len(self.positions) < 3:
            return self.rep_count

        # Detect direction
        avg_prev = sum(self.positions[:-1]) / (len(self.positions) - 1)
        direction_up = current_y < avg_prev

        # Detect rep on direction change
        if self.going_up is not None and self.going_up != direction_up:
            # Direction changed
            current_time = time.time()

            # Ensure minimum time between reps (0.5 sec) to avoid double counting
            if current_time - self.last_rep_time > 0.5:
                # Only count if movement exceeds threshold
                if (
                    abs(max(self.positions) - min(self.positions)) > 0.15
                ):  # Threshold for movement
                    self.rep_count += 1
                    self.last_rep_time = current_time

        self.going_up = direction_up
        return self.rep_count


def analyze_form(keypoints: List[Dict[str, Any]], exercise_type: str = "squat") -> str:
    """
    Simple form analysis based on keypoint positions

    Args:
        keypoints: List of keypoints from pose detection
        exercise_type: Type of exercise being performed

    Returns:
        Form assessment: "good", "check_knees", "check_back", "check_depth", etc.
    """
    # This is a simplified placeholder for form analysis
    if not keypoints or len(keypoints) < 10:  # Need at least basic keypoints
        return "unknown"

    # For this demo, just return a static result
    # In a real implementation, you would analyze the angles between joints
    return "good"


@websocket_router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time video processing

    Args:
        websocket: The WebSocket connection
        client_id: Client identifier
    """
    # Log connection attempt
    logger.info(f"WebSocket connection attempt from client: {client_id}")

    # Check CORS before accepting connection
    if not await cors_validation(websocket):
        logger.warning(f"CORS validation failed for client: {client_id}")
        return

    # Accept the connection
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for client: {client_id}")

    # Register the connection with the manager
    await manager.connect(websocket, client_id)

    # Set up a heartbeat task
    heartbeat_task = asyncio.create_task(manager.heartbeat(client_id))

    # Initialize session data for this client if it doesn't exist
    if client_id not in session_data:
        session_data[client_id] = {
            "rep_counter": RepCounter(),
            "exercise_type": "squat",  # Default exercise type
            "last_frame_time": time.time(),
            "processed_frames": 0,
        }

    try:
        # Send a welcome message to confirm connection
        await websocket.send_json(
            {"status": "connected", "message": "Connection established"}
        )

        while True:
            # Receive frame from client
            data = await websocket.receive_text()

            # Check if it's a control message (JSON)
            if data.startswith("{"):
                try:
                    control_data = json.loads(data)

                    # Handle exercise type change
                    if "exercise_type" in control_data:
                        session_data[client_id]["exercise_type"] = control_data[
                            "exercise_type"
                        ]
                        session_data[client_id][
                            "rep_counter"
                        ] = RepCounter()  # Reset counter

                    # Handle reset command
                    if control_data.get("action") == "reset":
                        session_data[client_id]["rep_counter"] = RepCounter()

                    await websocket.send_json({"status": "ok"})
                    continue
                except json.JSONDecodeError:
                    pass  # Not a valid JSON, treat as image data

            # Update frame timing statistics
            current_time = time.time()
            elapsed = current_time - session_data[client_id]["last_frame_time"]
            session_data[client_id]["last_frame_time"] = current_time
            session_data[client_id]["processed_frames"] += 1

            # Log occasional statistics
            if session_data[client_id]["processed_frames"] % 100 == 0:
                logger.info(
                    f"Client {client_id}: Processed {session_data[client_id]['processed_frames']} frames, {1/elapsed:.2f} FPS"
                )

            # Call the inference service
            analysis_results = await call_inference_service(data)

            # Handle errors from inference service
            if "error" in analysis_results:
                await websocket.send_json(analysis_results)
                continue

            # Extract keypoints and update rep counter
            keypoints = []
            if "keypoints" in analysis_results:
                keypoints = analysis_results["keypoints"]
                rep_count = session_data[client_id]["rep_counter"].update(keypoints)
                analysis_results["rep_count"] = rep_count

            # Analyze form if we have keypoints
            if keypoints:
                form_quality = analyze_form(
                    keypoints, session_data[client_id]["exercise_type"]
                )
                analysis_results["form_quality"] = form_quality

            # Send results back to client
            await websocket.send_json(analysis_results)

    except WebSocketDisconnect:
        # Clean up on disconnect
        logger.info(
            f"Client {client_id} disconnected after processing {session_data[client_id]['processed_frames']} frames"
        )
        heartbeat_task.cancel()
        manager.disconnect(client_id)
        # Keep session data for a while in case client reconnects
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
        heartbeat_task.cancel()
        manager.disconnect(client_id)


@websocket_router.websocket("/ws-simple/{client_id}")
async def simple_websocket(websocket: WebSocket, client_id: str):
    """Simple WebSocket test endpoint with minimal code"""
    logger.info(f"Simple WebSocket connection attempt from client: {client_id}")
    logger.info(f"Headers: {dict(websocket.headers)}")
    logger.info(f"Client host: {websocket.client.host}, port: {websocket.client.port}")
    logger.info(f"Connection type: {websocket.scope.get('type')}")
    logger.info(f"Connection path: {websocket.scope.get('path')}")

    try:
        # Accept the connection without any validation
        await websocket.accept()
        logger.info(f"Simple WebSocket connection accepted for client: {client_id}")

        # Send a welcome message
        await websocket.send_text("Connected to simple test endpoint")

        # Simple echo service
        while True:
            try:
                data = await websocket.receive_text()
                logger.info(f"Received from {client_id}: {data}")
                await websocket.send_text(f"Echo: {data}")
            except WebSocketDisconnect:
                logger.info(f"Simple WebSocket disconnected: {client_id}")
                break
            except Exception as e:
                logger.error(f"Simple WebSocket error: {str(e)}")
                break
    except Exception as e:
        logger.error(f"Simple WebSocket connection error: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
