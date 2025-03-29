from pydantic import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings"""

    # Base settings
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]

    # Inference service settings
    INFERENCE_SERVICE_URL: str = "http://localhost:8000"

    # WebSocket settings
    WS_HEARTBEAT_INTERVAL: int = 30  # seconds

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
