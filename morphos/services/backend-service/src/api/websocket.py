from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
import logging
import json
import base64
import io
import cv2
import numpy as np
import asyncio
from typing import Dict, Any

from core.managers import ConnectionManager
from core.security import verify_token

logger = logging.getLogger("morphos-websocket")
websocket_router = APIRouter()

# Initialize the connection manager
manager = ConnectionManager()


async def process_frame(frame_data: np.ndarray, client_id: str) -> Dict[str, Any]:
    """
    Process a video frame through ML models.
    This will be expanded to call the Dynamo inference service.

    Args:
        frame_data: The video frame as a numpy array
        client_id: The client ID for the connection

    Returns:
        Dict containing analysis results
    """
    # TODO: Implement actual call to Dynamo inference service
    # For now return mock data
    return {
        "keypoints": [{"x": 100, "y": 100}, {"x": 150, "y": 150}],  # Example keypoints
        "emotion": "focused",
        "rep_count": 0,
        "form_quality": "good",
    }


@websocket_router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str, token: str = None):
    """
    WebSocket endpoint for real-time video processing

    Args:
        websocket: The WebSocket connection
        client_id: Client identifier
        token: Authentication token (optional for now)
    """
    # Accept the connection
    await manager.connect(websocket, client_id)

    # Set up a heartbeat task
    heartbeat_task = asyncio.create_task(manager.heartbeat(client_id))

    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_text()

            # Process the received frame (base64 encoded image)
            try:
                # Decode base64 image
                encoded_data = data.split(",")[1] if "," in data else data
                frame_bytes = base64.b64decode(encoded_data)
                frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

                # Process the frame
                analysis_results = await process_frame(frame, client_id)

                # Send results back to client
                await manager.send_message(client_id, analysis_results)

            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        # Clean up on disconnect
        heartbeat_task.cancel()
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        heartbeat_task.cancel()
        manager.disconnect(client_id)
