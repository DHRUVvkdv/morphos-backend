from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio

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


# Simple connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(
            f"Client {client_id} connected. Total: {len(self.active_connections)}"
        )

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(
                f"Client {client_id} disconnected. Total: {len(self.active_connections)}"
            )

    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)


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


# WebSocket endpoint
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Echo back any received data (for testing)
            data = await websocket.receive_text()
            await websocket.send_text(f"You sent: {data}")
            logger.info(f"Received message from {client_id}: {data[:50]}...")
    except WebSocketDisconnect:
        manager.disconnect(client_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
