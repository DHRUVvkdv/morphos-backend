# Create a new file: morphos/services/backend-service/src/core/api_key.py

from fastapi import Request, HTTPException, status
import os
import logging
import sys
from typing import Optional, List

logger = logging.getLogger("morphos-api-key")


# Comprehensive logging of environment and API key
def log_environment_details():
    """Log detailed information about the environment"""
    logger.info("--- Comprehensive Environment Investigation ---")

    # Python interpreter and path details
    logger.info(f"Python Executable: {sys.executable}")
    logger.info(f"Python Path: {sys.path}")

    # Current working directory
    logger.info(f"Current Working Directory: {os.getcwd()}")

    # Environment variables investigation
    logger.info("--- Environment Variables Containing 'API_KEY' ---")
    api_key_vars = {
        key: value for key, value in os.environ.items() if "API_KEY" in key.upper()
    }

    for key, value in api_key_vars.items():
        logger.info(f"{key}: {'*' * len(value)}")  # Mask the actual value

    # Specific checks for API_KEY
    logger.info("--- Specific API_KEY Checks ---")
    logger.info(f"os.environ.get('API_KEY'): {os.environ.get('API_KEY')}")
    logger.info(f"'API_KEY' in os.environ: {'API_KEY' in os.environ}")


# Call the logging function
log_environment_details()


# Get API key from environment variable
def get_api_key():
    """Get API key from environment, ensure it's only retrieved when needed"""
    return os.environ.get("API_KEY")


EXEMPT_PATHS = ["/", "/health", "/docs", "/redoc", "/openapi.json"]


async def verify_api_key(request: Request) -> None:
    # Check if this is a WebSocket request by looking at the scope type
    if request.scope.get("type") == "websocket":
        logger.info("Skipping API key verification for WebSocket connection")
        return

    # Skip API key verification for exempt paths
    if request.url.path in EXEMPT_PATHS or request.url.path.startswith("/static"):
        return

    # Skip API key verification for WebSocket paths
    if request.url.path.startswith("/ws") or request.url.path.startswith("/ws-simple"):
        logger.info(
            f"Skipping API key verification for WebSocket path: {request.url.path}"
        )
        return

    # Skip OPTIONS requests (for CORS preflight)
    if request.method == "OPTIONS":
        return

    # Get API key only when needed
    api_key = get_api_key()

    # Ultra-verbose logging
    logger.info("--- API Key Verification ---")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request path: {request.url.path}")

    # Log all request headers for debugging
    for name, value in request.headers.items():
        logger.info(f"{name}: {value}")

    # Try multiple header variations
    api_key_header = (
        request.headers.get("X-API-Key")
        or request.headers.get("API-Key")
        or request.headers.get("x-api-key")
    )

    # Log API key details with maximum verbosity
    # logger.info(f"Effective API_KEY from environment: {api_key}")
    # logger.info(f"Received API key header: {api_key_header}")

    if not api_key_header:
        logger.warning(f"Missing API key for request to {request.url.path}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include X-API-Key header with your request.",
        )

    if api_key_header != api_key:
        logger.warning(f"Invalid API key for request to {request.url.path}")
        logger.warning(f"Expected key: {api_key}")
        logger.warning(f"Received key: {api_key_header}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # If we reach here, the API key is valid
    logger.info(f"Valid API key for request to {request.url.path}")

    # If we reach here, the API key is valid
    logger.info(f"Valid API key for request to {request.url.path}")
