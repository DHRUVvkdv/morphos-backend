# Create a new file: morphos/services/backend-service/src/core/api_key.py

from fastapi import Request, HTTPException, status
import os
import logging
from typing import Optional, List

logger = logging.getLogger("morphos-api-key")

# Get API key from environment variable or use a default one for development
API_KEY = os.environ.get("API_KEY")
EXEMPT_PATHS = ["/", "/health", "/docs", "/redoc", "/openapi.json"]


async def verify_api_key(request: Request) -> None:
    """
    Middleware to verify API key in header.

    Raises HTTPException if API key is invalid or missing.
    """
    # Skip API key verification for exempt paths
    if request.url.path in EXEMPT_PATHS or request.url.path.startswith("/static"):
        return

    # Skip API key verification for WebSocket connections
    if request.url.path.startswith("/ws/"):
        return

    # Skip OPTIONS requests (for CORS preflight)
    if request.method == "OPTIONS":
        return

    api_key_header = request.headers.get("X-API-Key")

    if not api_key_header:
        logger.warning(f"Missing API key for request to {request.url.path}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include X-API-Key header with your request.",
        )

    if api_key_header != API_KEY:
        logger.warning(f"Invalid API key for request to {request.url.path}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # If we reach here, the API key is valid
    logger.debug(f"Valid API key for request to {request.url.path}")
