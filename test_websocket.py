from fastapi import FastAPI, WebSocket
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test-websocket")

app = FastAPI()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    logger.info(f"Connection attempt from client: {client_id}")
    await websocket.accept()
    logger.info(f"Connection accepted for client: {client_id}")
    await websocket.send_text("Hello, client!")

    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received: {data}")
            await websocket.send_text(f"Echo: {data}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")


@app.get("/")
async def root():
    return {"message": "WebSocket test server"}


if __name__ == "__main__":
    uvicorn.run("test_websocket:app", host="0.0.0.0", port=8000, log_level="info")
