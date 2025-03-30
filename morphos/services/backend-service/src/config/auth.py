from pydantic_settings import BaseSettings
from typing import Optional, List
import os
import logging

logger = logging.getLogger("morphos-auth")


class Auth0Settings:
    """Auth0 Authentication settings without Pydantic"""

    def __init__(self):
        # Get domain from environment and ensure it has the protocol
        domain = os.environ.get("AUTH0_DOMAIN", "")
        if domain and not domain.startswith(("http://", "https://")):
            domain = f"https://{domain}"

        self.DOMAIN = domain

        # Get audience and ensure it has the protocol if it's a URL
        audience = os.environ.get("AUTH0_AUDIENCE", "")
        if (
            audience
            and not audience.startswith(("http://", "https://"))
            and "://" not in audience
        ):
            # Only add protocol if it looks like a domain rather than a unique identifier
            if "." in audience:
                audience = f"https://{audience}"

        self.AUDIENCE = audience

        self.CLIENT_ID = os.environ.get("AUTH0_CLIENT_ID", "")
        self.CLIENT_SECRET = os.environ.get("AUTH0_CLIENT_SECRET", "")
        self.ALGORITHMS = ["RS256"]


# Instantiate settings
auth0_settings = Auth0Settings()

# Print the loaded settings
logger.info(f"Loaded Auth0 settings:")
logger.info(f"DOMAIN: '{auth0_settings.DOMAIN}'")
logger.info(f"AUDIENCE: '{auth0_settings.AUDIENCE}'")
logger.info(f"CLIENT_ID: '{auth0_settings.CLIENT_ID[:5]}...' (truncated)")
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
