from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings"""

    # Base settings
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    # CORS settings
    CORS_ORIGINS: List[str] = [
        "*",  # Allow all origins for development
        "https://localhost:8080",  # Local development
        "http://localhost:8080",
        "http://localhost:3000",
        "https://localhost:3000",
        "null",  # Allow file:// protocol
        "https://morphos-backend-service-1020595365432.us-central1.run.app",  # Cloud Run service
    ]

    # Inference service settings
    INFERENCE_SERVICE_URL: str = "http://localhost:8000"

    # WebSocket settings
    WS_HEARTBEAT_INTERVAL: int = 30  # seconds

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
