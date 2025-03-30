from pydantic_settings import BaseSettings
from typing import Optional, List
import os
import logging

logger = logging.getLogger("morphos-auth")


class Auth0Settings:
    """Auth0 Authentication settings without Pydantic"""

    def __init__(self):
        self.DOMAIN = os.environ.get("AUTH0_DOMAIN", "")
        self.AUDIENCE = os.environ.get("AUTH0_AUDIENCE", "")
        self.CLIENT_ID = os.environ.get("AUTH0_CLIENT_ID", "")
        self.CLIENT_SECRET = os.environ.get("AUTH0_CLIENT_SECRET", "")
        self.ALGORITHMS = ["RS256"]


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
