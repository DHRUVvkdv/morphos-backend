from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any
import logging

from core.models.user import UserCreate, UserProfile, UserLogin
from core.auth import create_auth0_user, custom_signin
from core.db_operations import create_user, get_user_by_email

logger = logging.getLogger("morphos-auth")

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/signup", response_model=Dict[str, Any])
async def signup(user_data: UserCreate):
    """
    Register a new user in Auth0 and store profile in MongoDB with all fitness data
    """
    try:
        # First, check if user already exists in MongoDB
        existing_user = await get_user_by_email(user_data.email)
        if (
            existing_user is not None
        ):  # Changed from "if existing_user:" to "if existing_user is not None:"
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User with this email already exists",
            )

        # Create user in Auth0
        auth0_user = await create_auth0_user(
            user_data.email, user_data.password, user_data.name
        )

        # Prepare user data for MongoDB
        mongo_user_data = {
            "email": user_data.email,
            "name": user_data.name,
            "auth0_id": auth0_user["user_id"],
        }

        # Add physical stats if provided
        if user_data.height is not None:
            mongo_user_data["height"] = user_data.height
        if user_data.weight is not None:
            mongo_user_data["weight"] = user_data.weight
        if user_data.age is not None:
            mongo_user_data["age"] = user_data.age

        # Add fitness information if provided
        if user_data.fitness_level:
            mongo_user_data["fitness_level"] = user_data.fitness_level.value
        if user_data.workout_duration is not None:
            mongo_user_data["workout_duration"] = user_data.workout_duration
        if user_data.workout_frequency is not None:
            mongo_user_data["workout_frequency"] = user_data.workout_frequency
        if user_data.fitness_goals:
            mongo_user_data["fitness_goals"] = user_data.fitness_goals
        if user_data.available_equipment:
            mongo_user_data["available_equipment"] = user_data.available_equipment

        # Store in MongoDB
        db_user = await create_user(mongo_user_data)
        if db_user is None:  # Changed from "if not db_user:" to "if db_user is None:"
            logger.error("Failed to create user in MongoDB")
            # User was created in Auth0 but not in MongoDB
            # In a production app, you might want to delete the Auth0 user
            # or implement a cleanup/recovery mechanism

        # Return success response
        return {
            "status": "success",
            "message": "User created successfully",
            "user_id": auth0_user["user_id"],
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating user: {str(e)}",
        )


@router.post("/signin", response_model=Dict[str, Any])
async def signin(user_data: UserLogin):
    """
    Authenticate user and return JWT token
    """
    try:
        # Authenticate with Auth0
        token_data = await custom_signin(user_data.email, user_data.password)

        # Get user from MongoDB
        user = await get_user_by_email(user_data.email)

        # Return token and basic user info
        response = {
            "access_token": token_data["access_token"],
            "token_type": token_data["token_type"],
            "expires_in": token_data.get("expires_in", 3600),
        }

        # Add user info if available
        if user is not None:  # Changed from "if user:" to "if user is not None:"
            if "_id" in user:
                user["_id"] = str(user["_id"])
            response["user"] = user

        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication error: {str(e)}",
        )
