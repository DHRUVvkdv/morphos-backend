from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
import httpx
from typing import Dict, Any, Optional
import logging

from config.auth import auth0_settings
from core.auth import get_auth0_public_keys

# Configure logging
logger = logging.getLogger("morphos-security")

# OAuth2 scheme for token extraction - remove the unsupported parameter
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    Get the current authenticated user from JWT token
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        # Get Auth0 public keys for JWT verification
        jwks = await get_auth0_public_keys()

        # Decode token without verification first to get the key ID (kid)
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")

        if not kid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token header",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Find the right key
        rsa_key = {}
        for key in jwks:
            if key["kid"] == kid:
                rsa_key = {
                    "kty": key["kty"],
                    "kid": key["kid"],
                    "use": key["use"],
                    "n": key["n"],
                    "e": key["e"],
                }
                break

        if not rsa_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Key not found",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Verify the token
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=auth0_settings.ALGORITHMS,
            audience=auth0_settings.AUDIENCE,
            issuer=f"https://{auth0_settings.DOMAIN}/",
        )

        return payload

    except JWTError as e:
        logger.error(f"JWT validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Add a simple verification function for testing
async def verify_token(token: Optional[str] = None) -> bool:
    """
    Verify authentication token.
    This is a placeholder for actual token verification.

    Args:
        token: Authentication token

    Returns:
        True if token is valid, False otherwise
    """
    # Don't use "if token:" as it could be a database object
    # Instead use explicit check against None
    return token is not None
