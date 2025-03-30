from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
import os
import json
from api.auth_routes import router as auth_router
from core.database import init_db
from dotenv import load_dotenv
import logging
import pathlib


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("morphos-main")

# Get absolute path to current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Try to load .env from multiple possible locations
possible_env_paths = [
    os.path.join(current_dir, ".env"),
    os.path.join(parent_dir, ".env"),
    ".env",
]

env_loaded = False
for env_path in possible_env_paths:
    if os.path.exists(env_path):
        logger.info(f"Loading .env from: {env_path}")
        load_dotenv(env_path)
        env_loaded = True
        break

if not env_loaded:
    logger.warning("No .env file found in any expected location")

# Print all environment variables for debugging
logger.info("=== Environment Variables ===")
for key, value in os.environ.items():
    if key.startswith("AUTH0_"):
        if "SECRET" in key:
            logger.info(f"{key}: {value[:3]}... (truncated)")
        else:
            logger.info(f"{key}: {value}")

init_db()


from core.managers import ConnectionManager
from api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("morphos-api")

# Create FastAPI app
app = FastAPI(
    title="Morphos API Service",
    description="AI Workout Analysis API and WebSocket Service",
    version="0.1.0",
)

app.include_router(auth_router)

# Configure CORS - simple approach
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Initialize connection manager
manager = ConnectionManager()


# Basic health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}


# Root endpoint
@app.get("/")
async def root():
    return {
        "service": "Morphos API Service",
        "status": "running",
        "version": "0.1.0",
        "websocket_endpoint": "/ws/{client_id}",
    }


# Simple WebSocket endpoint without complex validation
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    # Accept the connection immediately
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for client: {client_id}")

    # Register with manager - don't call accept again
    await manager.connect(websocket, client_id)

    # Start heartbeat task in the background
    heartbeat_task = asyncio.create_task(manager.heartbeat(client_id, interval=15))

    try:
        # Send welcome message
        await websocket.send_json(
            {"status": "connected", "message": "Connection established"}
        )

        while True:
            # Receive data from client
            data = await websocket.receive_text()

            # Echo back for now as a test
            await websocket.send_json(
                {
                    "status": "ok",
                    "received_data_length": len(data),
                    "message": "Data received successfully",
                }
            )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for client {client_id}")
        manager.disconnect(client_id)
        heartbeat_task.cancel()
    except Exception as e:
        logger.error(f"Error in WebSocket for client {client_id}: {str(e)}")
        manager.disconnect(client_id)
        heartbeat_task.cancel()


if __name__ == "__main__":
    import uvicorn

    # Get port from environment variable
    port = int(os.environ.get("PORT", 8080))

    # Log startup
    logger.info(f"Starting server on port {port}")

    # Run with explicit WebSocket support
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        ws="websockets",  # Explicitly use the websockets implementation
        log_level="info",
    )
