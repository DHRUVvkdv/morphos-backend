from fastapi import WebSocket
import logging
import asyncio
import time
from typing import Dict, Any

logger = logging.getLogger("morphos-websocket")


class ConnectionManager:
    """Manager for WebSocket connections"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.last_activity: Dict[str, float] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """Register a new WebSocket connection"""
        # REMOVED the websocket.accept() call here - FastAPI handles this automatically
        self.active_connections[client_id] = websocket
        self.last_activity[client_id] = time.time()
        logger.info(
            f"Client {client_id} connected. Total connections: {len(self.active_connections)}"
        )

    def disconnect(self, client_id: str):
        """Remove a WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]

        if client_id in self.last_activity:
            del self.last_activity[client_id]

        logger.info(
            f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}"
        )

    async def send_message(self, client_id: str, message: Dict[str, Any]):
        """Send a message to a specific client"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.send_json(message)
            self.last_activity[client_id] = time.time()

    async def broadcast(self, message: Dict[str, Any]):
        """Send a message to all connected clients"""
        for client_id, websocket in self.active_connections.items():
            await websocket.send_json(message)
            self.last_activity[client_id] = time.time()

    async def heartbeat(self, client_id: str, interval: int = 30):
        """Send periodic heartbeats to keep the connection alive"""
        try:
            while client_id in self.active_connections:
                await asyncio.sleep(interval)
                if client_id in self.active_connections:
                    await self.active_connections[client_id].send_json(
                        {"type": "heartbeat"}
                    )
                    logger.debug(f"Sent heartbeat to client {client_id}")
        except Exception as e:
            logger.error(f"Heartbeat error for client {client_id}: {str(e)}")
            self.disconnect(client_id)
