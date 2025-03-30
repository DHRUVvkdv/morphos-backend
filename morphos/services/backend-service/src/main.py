from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
import os
import json

# Import your existing manager
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

    # Register with manager
    await manager.connect(websocket, client_id)

    # Start heartbeat task in the background
    heartbeat_task = asyncio.create_task(manager.heartbeat(client_id))

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
