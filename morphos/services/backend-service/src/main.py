from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    Request,
    Depends,
    HTTPException,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.openapi.utils import get_openapi
import logging
import asyncio
import os
import json
from dotenv import load_dotenv
import sys
import traceback
from core.api_key import verify_api_key
from core.debug_api_key import debug_api_key_loading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("morphos-main")

# Load environment variables
load_dotenv()

debug_api_key_loading()


# Import routers and database after environment is set up
from api.auth_routes import router as auth_router
from api.routes import router as main_router
from api.profile_routes import router as profile_router
from core.database import init_db
from core.managers import ConnectionManager

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

# ====== ENVIRONMENT VALIDATION ======
# Validate critical environment variables
required_vars = [
    "AUTH0_DOMAIN",
    "AUTH0_CLIENT_ID",
    "AUTH0_CLIENT_SECRET",
    "AUTH0_AUDIENCE",
]

missing_vars = []
for var in required_vars:
    if not os.environ.get(var):
        missing_vars.append(var)

if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    logger.error(
        "Authentication features will not work properly without these variables"
    )
else:
    logger.info("All required Auth0 environment variables are set")

# Check Auth0 domain format
auth0_domain = os.environ.get("AUTH0_DOMAIN", "")
if auth0_domain and not auth0_domain.startswith(("http://", "https://")):
    if auth0_domain.strip():  # Make sure it's not just whitespace
        logger.info(f"Adding https:// protocol to AUTH0_DOMAIN: {auth0_domain}")
        os.environ["AUTH0_DOMAIN"] = f"https://{auth0_domain}"
    else:
        logger.error("AUTH0_DOMAIN is empty or contains only whitespace")

# Display the final Auth0 configuration (truncated for secrets)
logger.info("=== Auth0 Configuration ===")
logger.info(f"AUTH0_DOMAIN: {os.environ.get('AUTH0_DOMAIN', '')}")
logger.info(f"AUTH0_AUDIENCE: {os.environ.get('AUTH0_AUDIENCE', '')}")
logger.info(
    f"AUTH0_CLIENT_ID: {os.environ.get('AUTH0_CLIENT_ID', '')[:5] if os.environ.get('AUTH0_CLIENT_ID') else 'not set'}"
)
if os.environ.get("AUTH0_CLIENT_SECRET"):
    logger.info(
        f"AUTH0_CLIENT_SECRET: {os.environ.get('AUTH0_CLIENT_SECRET', '')[:5]}... (truncated)"
    )
else:
    logger.info("AUTH0_CLIENT_SECRET: not set")
# ====== END ENVIRONMENT VALIDATION ======

# Print environment variables for debugging (exclude secrets)
for key, value in os.environ.items():
    if key.startswith("MONGODB_"):
        logger.info(f"{key}: {'*****' if 'URI' in key else value}")

# Import routers and database after environment is set up
from api.auth_routes import router as auth_router
from api.routes import router as main_router
from api.profile_routes import router as profile_router
from core.database import init_db
from core.managers import ConnectionManager

# Initialize database
db = init_db()
if db is not None:
    logger.info("Database connection initialized successfully")

    # Ensure indexes for efficient queries
    try:
        db.users.create_index("email", unique=True)
        db.users.create_index("auth0_id", unique=True)
        logger.info("Database indexes created successfully")
    except Exception as e:
        logger.error(f"Error creating database indexes: {str(e)}")
else:
    logger.warning("Database initialization failed or disabled")

# Create security scheme
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Morphos API Service",
        version="0.1.0",
        description="AI Workout Analysis API and WebSocket Service",
        routes=app.routes,
    )

    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
    }

    # Set global security requirement
    openapi_schema["security"] = [{"APIKeyHeader": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Create FastAPI app
app = FastAPI(
    title="Morphos API Service",
    description="AI Workout Analysis API and WebSocket Service",
    version="0.1.0",
    # Add security definition to OpenAPI schema
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    dependencies=[Depends(verify_api_key)],
)

app.openapi = custom_openapi


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(main_router)
app.include_router(profile_router)  # Add profile routes

# Initialize connection manager
manager = ConnectionManager()


# Basic health check endpoint
@app.get("/health")
async def health_check():
    # Check database connection
    db_status = "connected" if db is not None else "disconnected"

    # Additional health metrics
    mongo_status = "ok"
    if db is not None:
        try:
            # Ping MongoDB to verify connection
            db.command("ping")
        except Exception as e:
            mongo_status = f"error: {str(e)}"

    return {
        "status": "ok",
        "database": db_status,
        "mongo_status": mongo_status,
        "version": "0.1.0",
        "auth0_domain": os.environ.get("AUTH0_DOMAIN", "not set"),
        "api_key_enabled": True,
        "api_key_value": (
            "****" + os.environ.get("API_KEY")[-4:]
            if os.environ.get("API_KEY")
            else "not set"
        ),
    }


# Root endpoint
@app.get("/")
async def root():
    return {
        "service": "Morphos API Service",
        "status": "running",
        "version": "0.1.0",
        "websocket_endpoint": "/ws/{client_id}",
        "database_connected": db is not None,
    }


# WebSocket endpoint
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    # Accept the connection
    try:
        await websocket.accept()
        logger.info(f"WebSocket connection accepted for client: {client_id}")

        # Register with manager
        await manager.connect(websocket, client_id)

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(manager.heartbeat(client_id, interval=15))

        try:
            # Send welcome message
            await websocket.send_json(
                {"status": "connected", "message": "Connection established"}
            )

            while True:
                # Receive data from client
                data = await websocket.receive_text()

                # Handle JSON messages separately
                if data.startswith("{"):
                    try:
                        json_data = json.loads(data)
                        # Process JSON message (could be control commands)
                        await websocket.send_json(
                            {
                                "status": "ok",
                                "type": "json_response",
                                "message": "JSON data received and processed",
                            }
                        )
                        continue
                    except json.JSONDecodeError:
                        pass  # Not valid JSON, process as regular message

                # Echo back for video/binary data
                await websocket.send_json(
                    {
                        "status": "ok",
                        "type": "data_received",
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
            logger.error(traceback.format_exc())
            manager.disconnect(client_id)
            heartbeat_task.cancel()
    except Exception as e:
        logger.error(f"Error in websocket_endpoint for client {client_id}: {str(e)}")
        logger.error(traceback.format_exc())


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
