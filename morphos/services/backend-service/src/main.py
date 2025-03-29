from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
import os

# Import your existing ConnectionManager
from core.managers import ConnectionManager

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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize connection manager
manager = ConnectionManager()


# REST API endpoint
@app.get("/")
async def root():
    return {"service": "Morphos API Service", "status": "running", "version": "0.1.0"}


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}


# WebSocket endpoint with improved Cloud Run handling
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)

    # Start heartbeat task in the background
    heartbeat_task = asyncio.create_task(manager.heartbeat(client_id, interval=30))

    try:
        while True:
            # Use a shorter timeout than Cloud Run's 60s default
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=50)
                await websocket.send_text(f"You sent: {data}")
                logger.info(f"Received message from {client_id}: {data[:50]}...")
            except asyncio.TimeoutError:
                # The heartbeat task will handle sending pings
                pass
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
    # Note: disable reload in production (Cloud Run)
    is_dev = os.environ.get("ENVIRONMENT", "").lower() != "production"

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        ws="websockets",  # Explicitly use the websockets implementation
        log_level="info",
        timeout_keep_alive=70,  # Helps with WebSocket connections
        reload=is_dev,  # Only use reload in development
    )
