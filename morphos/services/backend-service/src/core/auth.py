from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
import httpx
import time
from typing import Dict, Optional, List
from pydantic import BaseModel, EmailStr, Field
import logging
import json

# Add this import
from jose import jwk

from config.auth import auth0_settings

logger = logging.getLogger("morphos-auth")


# Models
class TokenData(BaseModel):
    sub: str
    exp: int
    azp: str
    iss: str


class UserBase(BaseModel):
    email: EmailStr


class UserCreate(UserBase):
    password: str
    name: Optional[str] = None


class UserLogin(UserBase):
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int


class UserProfile(UserBase):
    user_id: str = Field(..., alias="sub")
    name: Optional[str] = None


# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Cache for Auth0 public key
JWKS_CACHE = {"keys": None, "expires_at": 0}


async def get_auth0_public_keys():
    """Get Auth0 public keys for JWT verification"""
    # Use cached keys if available and not expired
    if JWKS_CACHE["keys"] and JWKS_CACHE["expires_at"] > time.time():
        return JWKS_CACHE["keys"]

    # Fetch keys from Auth0
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://{auth0_settings.DOMAIN}/.well-known/jwks.json"
        )
        if response.status_code != 200:
            logger.error(f"Failed to get Auth0 public keys: {response.text}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get authentication keys",
            )

        keys = response.json()["keys"]
        # Cache for 6 hours
        JWKS_CACHE["keys"] = keys
        JWKS_CACHE["expires_at"] = time.time() + 6 * 3600

        return keys


async def get_token(email: str, password: str) -> TokenResponse:
    """Get Auth0 token using Resource Owner Password flow"""
    # Log all relevant configuration
    logger.info("=== Auth0 Configuration ===")
    logger.info(f"DOMAIN: '{auth0_settings.DOMAIN}'")
    logger.info(f"CLIENT_ID: '{auth0_settings.CLIENT_ID}'")
    logger.info(f"CLIENT_SECRET: '{auth0_settings.CLIENT_SECRET[:3]}...' (truncated)")
    logger.info(f"AUDIENCE: '{auth0_settings.AUDIENCE}'")

    # Check if Auth0 is configured
    if not auth0_settings.DOMAIN:
        logger.error("AUTH0_DOMAIN is empty")
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Authentication service not configured. Please set up Auth0 credentials.",
        )

    if not auth0_settings.CLIENT_ID:
        logger.error("AUTH0_CLIENT_ID is empty")
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Authentication service not configured. Please set up Auth0 credentials.",
        )

    if not auth0_settings.CLIENT_SECRET:
        logger.error("AUTH0_CLIENT_SECRET is empty")
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Authentication service not configured. Please set up Auth0 credentials.",
        )

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://{auth0_settings.DOMAIN}/oauth/token",
            data={
                "grant_type": "password",
                "username": email,
                "password": password,
                "client_id": auth0_settings.CLIENT_ID,
                "client_secret": auth0_settings.CLIENT_SECRET,
                "audience": auth0_settings.AUDIENCE,
                "scope": "openid profile email",
            },
        )

        if response.status_code != 200:
            error_msg = response.json().get(
                "error_description", "Authentication failed"
            )
            logger.error(f"Auth0 token error: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=error_msg
            )

        token_data = response.json()
        return TokenResponse(
            access_token=token_data["access_token"],
            token_type=token_data["token_type"],
            expires_in=token_data["expires_in"],
        )


async def create_auth0_user(email: str, password: str, name: Optional[str] = None):
    """Create a new user in Auth0"""
    # Check if Auth0 is configured
    if (
        not auth0_settings.DOMAIN
        or not auth0_settings.CLIENT_ID
        or not auth0_settings.CLIENT_SECRET
    ):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Authentication service not configured. Please set up Auth0 credentials.",
        )
    # First, get Management API token
    mgmt_token = await get_management_token()

    # Create user
    async with httpx.AsyncClient() as client:
        user_data = {
            "email": email,
            "password": password,
            "connection": "Username-Password-Authentication",
            "email_verified": False,
        }

        if name:
            user_data["name"] = name

        response = await client.post(
            f"https://{auth0_settings.DOMAIN}/api/v2/users",
            headers={"Authorization": f"Bearer {mgmt_token}"},
            json=user_data,
        )

        if response.status_code not in (200, 201):
            error_msg = response.json().get("message", "Failed to create user")
            logger.error(f"Create user error: {error_msg}")

            # Check if user already exists
            if "already exists" in error_msg.lower():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT, detail="User already exists"
                )

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg
            )

        return response.json()


async def get_management_token() -> str:
    """Get Auth0 Management API token"""
    # Check if Auth0 is configured
    if (
        not auth0_settings.DOMAIN
        or not auth0_settings.CLIENT_ID
        or not auth0_settings.CLIENT_SECRET
    ):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Authentication service not configured. Please set up Auth0 credentials.",
        )
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://{auth0_settings.DOMAIN}/oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": auth0_settings.CLIENT_ID,
                "client_secret": auth0_settings.CLIENT_SECRET,
                "audience": f"https://{auth0_settings.DOMAIN}/api/v2/",
            },
        )

        if response.status_code != 200:
            logger.error(f"Failed to get management token: {response.text}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get management credentials",
            )

        return response.json()["access_token"]


async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserProfile:
    """Validate JWT token and return user info"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Decode token using Auth0 public key
        signing_key = await jwk_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            key=signing_key.to_pem().decode("utf-8"),
            algorithms=auth0_settings.ALGORITHMS,
            audience=auth0_settings.AUDIENCE,
            issuer=f"https://{auth0_settings.DOMAIN}/",
        )

        # Extract user info
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception

        # Get user profile from Auth0
        return UserProfile(
            sub=user_id, email=payload.get("email", ""), name=payload.get("name")
        )

    except JWTError as e:
        logger.error(f"JWT error: {str(e)}")
        raise credentials_exception
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise credentials_exception


class JWKClient:
    def __init__(self, domain):
        self.domain = domain
        self.cache = {}

    async def get_jwks(self):
        if "jwks" in self.cache and self.cache["exp"] > time.time():
            return self.cache["jwks"]

        async with httpx.AsyncClient() as client:
            url = f"https://{self.domain}/.well-known/jwks.json"
            response = await client.get(url)

            if response.status_code != 200:
                raise Exception(f"Failed to fetch JWKS: {response.status_code}")

            jwks = response.json()
            self.cache["jwks"] = jwks
            self.cache["exp"] = time.time() + 3600  # Cache for 1 hour

            return jwks

    async def get_signing_key(self, kid):
        jwks = await self.get_jwks()

        for key in jwks["keys"]:
            if key["kid"] == kid:
                return key

        raise Exception(f"Unable to find key with kid {kid}")

    async def get_signing_key_from_jwt(self, token):
        # Get the kid from the token header
        try:
            headers = jwt.get_unverified_header(token)
            kid = headers["kid"]
            key = await self.get_signing_key(kid)

            # Convert to PEM format
            pem_key = jwk.construct(key)

            return pem_key
        except Exception as e:
            logger.error(f"Error getting signing key: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token header"
            )


# Initialize JWK client
jwk_client = JWKClient(auth0_settings.DOMAIN)
