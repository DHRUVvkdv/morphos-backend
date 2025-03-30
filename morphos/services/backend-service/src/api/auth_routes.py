from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from typing import Optional

from core.auth import (
    UserCreate,
    UserLogin,
    get_token,
    create_auth0_user,
    custom_signin,
)
from core.database import get_db

router = APIRouter(prefix="/auth", tags=["authentication"])


class UserResponse(BaseModel):
    email: EmailStr
    name: Optional[str] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


@router.post("/signup")
async def signup(user_data: UserCreate):
    """Register a new user"""
    try:
        # Create user in Auth0
        auth0_user = await create_auth0_user(
            user_data.email, user_data.password, user_data.name
        )

        # Store in MongoDB (if enabled)
        db = get_db()
        if db:
            user_doc = {
                "email": user_data.email,
                "name": user_data.name,
                "auth0_id": auth0_user["user_id"],
            }
            db.users.insert_one(user_doc)

        # Return success without trying to get a token
        return {
            "status": "success",
            "message": "User created successfully",
            "user_id": auth0_user["user_id"],
        }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating user: {str(e)}",
        )


@router.post("/signin", response_model=TokenResponse)
async def signin(user_data: UserLogin):
    """Authenticate user and return JWT token"""
    try:
        token_data = await custom_signin(user_data.email, user_data.password)
        return TokenResponse(
            access_token=token_data["access_token"],
            token_type=token_data["token_type"],
            expires_in=token_data["expires_in"],
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication error: {str(e)}",
        )
