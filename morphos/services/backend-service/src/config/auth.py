from pydantic import BaseSettings
from typing import Optional, List
import os
import logging

logger = logging.getLogger("morphos-auth")

# Debug log to see what environment variables are available
logger.info(f"Environment variables for Auth0:")
logger.info(f"AUTH0_DOMAIN: {os.environ.get('AUTH0_DOMAIN', 'Not set')}")
logger.info(f"AUTH0_CLIENT_ID: {os.environ.get('AUTH0_CLIENT_ID', 'Not set')}")
logger.info(f"AUTH0_CLIENT_SECRET: {os.environ.get('AUTH0_CLIENT_SECRET', 'Not set')}")
logger.info(f"AUTH0_AUDIENCE: {os.environ.get('AUTH0_AUDIENCE', 'Not set')}")


class Auth0Settings(BaseSettings):
    """Auth0 Authentication settings"""

    DOMAIN: str
    AUDIENCE: str
    CLIENT_ID: str
    CLIENT_SECRET: str
    ALGORITHMS: List[str] = ["RS256"]

    # MongoDB settings
    MONGODB_URI: Optional[str] = None
    MONGODB_DB_NAME: str = "morphos_db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = "AUTH0_"


# Instantiate settings
auth0_settings = Auth0Settings()

# Print the loaded settings
logger.info(f"Loaded Auth0 settings:")
logger.info(f"DOMAIN: '{auth0_settings.DOMAIN}'")
logger.info(f"AUDIENCE: '{auth0_settings.AUDIENCE}'")
logger.info(f"CLIENT_ID: '{auth0_settings.CLIENT_ID}'")
logger.info(f"CLIENT_SECRET: '{auth0_settings.CLIENT_SECRET[:5]}...' (truncated)")

# Log warning if values are missing
if (
    not auth0_settings.DOMAIN
    or not auth0_settings.CLIENT_ID
    or not auth0_settings.CLIENT_SECRET
):
    logger.warning(
        "Auth0 configuration missing. Authentication features will be limited."
    )
