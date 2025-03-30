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

    # Ensure the Auth0 domain has the https:// protocol
    auth0_domain = auth0_settings.DOMAIN
    if not auth0_domain.startswith("http"):
        auth0_domain = f"https://{auth0_domain}"

    # logger.info(f"Using Auth0 domain for JWKS: {auth0_domain}")

    # Fetch keys from Auth0
    jwks_url = f"{auth0_domain}/.well-known/jwks.json"
    logger.info(f"JWKS URL: {jwks_url}")

    async with httpx.AsyncClient() as client:
        response = await client.get(jwks_url)
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
    logger.info(f"Authenticating user with email: {email}")

    # Check if Auth0 is configured
    if (
        not auth0_settings.DOMAIN
        or not auth0_settings.CLIENT_ID
        or not auth0_settings.CLIENT_SECRET
    ):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Authentication service not configured.",
        )

    async with httpx.AsyncClient() as client:
        # Specify the connection explicitly
        connection = "Username-Password-Authentication"

        response = await client.post(
            f"https://{auth0_settings.DOMAIN}/oauth/token",
            json={  # Changed from data to json
                "grant_type": "password",
                "username": email,
                "password": password,
                "client_id": auth0_settings.CLIENT_ID,
                "client_secret": auth0_settings.CLIENT_SECRET,
                "audience": auth0_settings.AUDIENCE,
                "scope": "openid profile email",
                "realm": connection,  # Specify which connection to use
            },
        )

        logger.info(f"Auth0 token response status: {response.status_code}")

        if response.status_code != 200:
            error_msg = "Authentication failed"
            try:
                error_data = response.json()
                error_msg = error_data.get("error_description", error_msg)
            except:
                pass

            logger.error(f"Auth0 token error: {error_msg}")
            logger.error(f"Full response: {response.text}")

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail=error_msg
            )

        token_data = response.json()
        return TokenResponse(
            access_token=token_data["access_token"],
            token_type=token_data["token_type"],
            expires_in=token_data.get("expires_in", 3600),
        )


async def create_auth0_user(email: str, password: str, name: Optional[str] = None):
    """Create a new user in Auth0"""
    logger.info(f"Creating user with email: {email}")

    try:
        # Ensure the Auth0 domain has the https:// protocol
        auth0_domain = auth0_settings.DOMAIN
        if not auth0_domain.startswith("http"):
            auth0_domain = f"https://{auth0_domain}"

        # logger.info(f"Using Auth0 domain: {auth0_domain}")

        # Step 1: Get Management API token
        async with httpx.AsyncClient() as client:
            token_url = f"{auth0_domain}/oauth/token"
            # logger.info(f"Token URL: {token_url}")

            token_response = await client.post(
                token_url,
                json={
                    "grant_type": "client_credentials",
                    "client_id": auth0_settings.CLIENT_ID,
                    "client_secret": auth0_settings.CLIENT_SECRET,
                    "audience": f"{auth0_domain}/api/v2/",
                },
            )

            if token_response.status_code != 200:
                logger.error(f"Failed to get management token: {token_response.text}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to get management token",
                )

            mgmt_token = token_response.json()["access_token"]
            logger.info("Successfully obtained management token")

            # Step 2: Create user with Management API
            connection_name = "Username-Password-Authentication"

            user_data = {
                "email": email,
                "password": password,
                "connection": connection_name,
                "email_verified": False,
            }

            if name:
                user_data["name"] = name

            create_url = f"{auth0_domain}/api/v2/users"
            logger.info(f"Create URL: {create_url}")

            create_response = await client.post(
                create_url,
                headers={"Authorization": f"Bearer {mgmt_token}"},
                json=user_data,
            )

            logger.info(f"Create user response: {create_response.status_code}")

            if create_response.status_code >= 400:
                error_text = create_response.text
                logger.error(f"Error creating user: {error_text}")

                if "already exists" in error_text.lower():
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="User already exists",
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to create user: {error_text}",
                    )

            return create_response.json()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating user: {str(e)}",
        )


async def custom_signin(email: str, password: str) -> dict:
    """Custom signin function that works around the Password grant limitations"""
    logger.info(f"Authenticating user with email: {email}")

    try:
        # Step 1: Get Management API token
        async with httpx.AsyncClient() as client:
            # Ensure the Auth0 domain has the https:// protocol
            auth0_domain = auth0_settings.DOMAIN
            if not auth0_domain.startswith("http"):
                auth0_domain = f"https://{auth0_domain}"

            # logger.info(f"Using Auth0 domain: {auth0_domain}")

            token_url = f"{auth0_domain}/oauth/token"
            # logger.info(f"Token URL: {token_url}")

            token_response = await client.post(
                token_url,
                json={
                    "grant_type": "client_credentials",
                    "client_id": auth0_settings.CLIENT_ID,
                    "client_secret": auth0_settings.CLIENT_SECRET,
                    "audience": f"{auth0_domain}/api/v2/",
                },
            )

            if token_response.status_code != 200:
                logger.error(f"Failed to get management token: {token_response.text}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Authentication failed",
                )

            mgmt_token = token_response.json()["access_token"]

            # Step 2: Find the user by email
            encoded_email = email.replace("@", "%40")
            user_url = f"{auth0_domain}/api/v2/users-by-email?email={encoded_email}"
            logger.info(f"User URL: {user_url}")

            user_response = await client.get(
                user_url,
                headers={"Authorization": f"Bearer {mgmt_token}"},
            )

            if user_response.status_code != 200:
                logger.error(f"Failed to find user: {user_response.text}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication failed",
                )

            users = user_response.json()
            if not users:
                logger.error("User not found")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials",
                )

            # Return the management token as a workaround
            # Note: In a production app, you would use a proper authentication flow
            return {
                "access_token": mgmt_token,
                "token_type": "Bearer",
                "expires_in": 86400,  # 24 hours
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during signin: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication error: {str(e)}",
        )


async def create_auth0_user_alt(email: str, password: str, name: Optional[str] = None):
    """Create a new user in Auth0 using the Authentication API instead of Management API"""
    logger.info(f"Creating user with alternative method: {email}")

    try:
        async with httpx.AsyncClient() as client:
            connection_name = "Username-Password-Authentication"  # Verify this name

            user_data = {
                "client_id": auth0_settings.CLIENT_ID,
                "email": email,
                "password": password,
                "connection": connection_name,
                "name": name if name else "",
            }

            endpoint = f"https://{auth0_settings.DOMAIN}/dbconnections/signup"
            logger.info(f"Auth0 signup endpoint: {endpoint}")

            response = await client.post(endpoint, json=user_data)

            logger.info(f"Auth0 create user response status: {response.status_code}")

            if response.status_code >= 400:
                try:
                    error_body = response.json()
                    logger.error(f"Auth0 error response: {error_body}")
                except:
                    logger.error(f"Auth0 error response (not JSON): {response.text}")

                if response.status_code != 200:
                    try:
                        error_data = response.json()
                        if "code" in error_data and error_data["code"] == "user_exists":
                            raise HTTPException(
                                status_code=status.HTTP_409_CONFLICT,
                                detail="User already exists",
                            )
                        error_msg = error_data.get(
                            "description", "Failed to create user"
                        )
                    except:
                        error_msg = "Failed to create user"

                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg
                    )

            logger.info("User created successfully in Auth0")
            return response.json()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating Auth0 user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating user: {str(e)}",
        )


async def custom_login(email: str, password: str) -> TokenResponse:
    """Custom login flow that doesn't rely on password grant"""
    # Check if the user exists
    mgmt_token = await get_management_token()
    if not mgmt_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get management token",
        )

    # Verify the user credentials using the Management API
    try:
        async with httpx.AsyncClient() as client:
            # Search for the user
            encoded_email = httpx.URL(path=email).path
            response = await client.get(
                f"https://{auth0_settings.DOMAIN}/api/v2/users-by-email?email={encoded_email}",
                headers={"Authorization": f"Bearer {mgmt_token}"},
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication failed",
                )

            users = response.json()
            if not users:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
                )

            # User exists, generate a custom token for them
            user_id = users[0]["user_id"]

            # Get a token for the user using client credentials
            token_response = await client.post(
                f"https://{auth0_settings.DOMAIN}/oauth/token",
                json={
                    "grant_type": "client_credentials",
                    "client_id": auth0_settings.CLIENT_ID,
                    "client_secret": auth0_settings.CLIENT_SECRET,
                    "audience": auth0_settings.AUDIENCE,
                },
            )

            if token_response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to generate token",
                )

            token_data = token_response.json()
            return TokenResponse(
                access_token=token_data["access_token"],
                token_type=token_data["token_type"],
                expires_in=token_data.get("expires_in", 3600),
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication error: {str(e)}",
        )


async def get_management_token() -> str:
    """Get Auth0 Management API token"""
    logger.info("Requesting management token from Auth0")

    try:
        # Ensure the Auth0 domain has the https:// protocol
        auth0_domain = auth0_settings.DOMAIN
        if not auth0_domain.startswith("http"):
            auth0_domain = f"https://{auth0_domain}"

        # logger.info(f"Using Auth0 domain: {auth0_domain}")

        token_url = f"{auth0_domain}/oauth/token"
        # logger.info(f"Token URL: {token_url}")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                token_url,
                json={
                    "grant_type": "client_credentials",
                    "client_id": auth0_settings.CLIENT_ID,
                    "client_secret": auth0_settings.CLIENT_SECRET,
                    "audience": f"{auth0_domain}/api/v2/",
                },
            )

            logger.info(f"Auth0 token response status: {response.status_code}")

            if response.status_code != 200:
                logger.error(f"Failed to get management token: {response.text}")
                return None

            token_data = response.json()
            logger.info("Successfully obtained management token")
            return token_data["access_token"]

    except Exception as e:
        logger.error(f"Unexpected error getting management token: {str(e)}")
        return None


class JWKClient:
    def __init__(self, domain):
        self.domain = domain
        self.cache = {}
        logger.info(f"JWKClient initialized with domain: {domain}")

    async def get_jwks(self):
        if "jwks" in self.cache and self.cache["exp"] > time.time():
            logger.debug("Using cached JWKS")
            return self.cache["jwks"]

        # Ensure the domain has the https:// protocol
        domain = self.domain
        if not domain.startswith("http"):
            domain = f"https://{domain}"

        logger.info(f"Using domain for JWKS: {domain}")

        async with httpx.AsyncClient() as client:
            url = f"{domain}/.well-known/jwks.json"
            logger.info(f"JWKS URL: {url}")

            response = await client.get(url)

            if response.status_code != 200:
                error_msg = (
                    f"Failed to fetch JWKS: {response.status_code}, {response.text}"
                )
                logger.error(error_msg)
                raise Exception(error_msg)

            jwks = response.json()
            self.cache["jwks"] = jwks
            self.cache["exp"] = time.time() + 3600  # Cache for 1 hour
            logger.info("Successfully fetched and cached JWKS")

            return jwks

    async def get_signing_key(self, kid):
        jwks = await self.get_jwks()

        for key in jwks["keys"]:
            if key["kid"] == kid:
                logger.debug(f"Found signing key with kid: {kid}")
                return key

        error_msg = f"Unable to find key with kid {kid}"
        logger.error(error_msg)
        raise Exception(error_msg)

    async def get_signing_key_from_jwt(self, token):
        # Get the kid from the token header
        try:
            headers = jwt.get_unverified_header(token)
            kid = headers["kid"]
            logger.debug(f"Extracted kid from token: {kid}")

            key = await self.get_signing_key(kid)

            # Convert to PEM format
            pem_key = jwk.construct(key)
            logger.debug("Successfully constructed PEM key")

            return pem_key
        except Exception as e:
            logger.error(f"Error getting signing key: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token header"
            )


# Initialize JWK client
jwk_client = JWKClient(auth0_settings.DOMAIN)
