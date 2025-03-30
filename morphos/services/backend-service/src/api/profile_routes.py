from fastapi import APIRouter, HTTPException, status, Query, Body
from typing import Dict, Any, Optional, List
import logging
import sys
import traceback
from core.models.user import UserUpdate, UserProfile
from core.db_operations import (
    get_user_by_email,
    update_user_profile_by_email,
    update_user_achievements_by_email,
    get_leaderboard,
)

# Enhanced logging
logger = logging.getLogger("morphos-profile")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

router = APIRouter(prefix="/profile", tags=["profile"])


@router.get("/me", response_model=Dict[str, Any])
async def get_profile(email: str):
    """
    Get the user's profile with all fitness data by email
    """
    try:
        logger.info(f"Getting profile for user with email: {email}")

        if not email:
            logger.error("Email not provided")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Email is required"
            )

        # Get user from database
        user = await get_user_by_email(email)

        if user is None:
            logger.warning(f"User profile not found for email: {email}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found"
            )

        logger.info(f"Successfully retrieved profile for user with email: {email}")

        # Convert MongoDB _id to string for serialization
        if "_id" in user:
            user["_id"] = str(user["_id"])

        return user
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_profile: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving user profile: {str(e)}",
        )


@router.put("/me", response_model=Dict[str, Any])
async def update_profile(email: str, profile_data: UserUpdate):
    """
    Update the user's profile by email
    """
    try:
        logger.info(f"Updating profile for user with email: {email}")

        if not email:
            logger.error("Email not provided")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Email is required"
            )

        # Prepare update data - exclude None values
        update_data = profile_data.dict(exclude_unset=True, exclude_none=True)

        # Convert enum values to strings if present
        if "fitness_level" in update_data and update_data["fitness_level"]:
            update_data["fitness_level"] = update_data["fitness_level"].value

        # Update user in database
        updated_user = await update_user_profile_by_email(email, update_data)
        if updated_user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found or update failed",
            )

        logger.info(f"Successfully updated profile for user with email: {email}")

        # Convert MongoDB _id to string for serialization
        if "_id" in updated_user:
            updated_user["_id"] = str(updated_user["_id"])

        return updated_user
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in update_profile: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating user profile: {str(e)}",
        )


@router.post("/achievements", response_model=Dict[str, Any])
async def update_achievements(email: str, achievement_data: Dict[str, Any] = Body(...)):
    """
    Update user achievement metrics (workout streak, total workouts, etc.) by email

    Can be used to increment counters, add badges, etc.
    """
    try:
        logger.info(f"Updating achievements for user with email: {email}")

        if not email:
            logger.error("Email not provided")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Email is required"
            )

        # Update achievements in database
        updated_user = await update_user_achievements_by_email(email, achievement_data)
        if updated_user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found or update failed",
            )

        logger.info(f"Successfully updated achievements for user with email: {email}")

        # Convert MongoDB _id to string for serialization
        if "_id" in updated_user:
            updated_user["_id"] = str(updated_user["_id"])

        return updated_user
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in update_achievements: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating user achievements: {str(e)}",
        )


@router.get("/stats")
async def get_user_stats(email: str):
    """
    Get the user's workout statistics summary by email
    """
    try:
        logger.info(f"Getting stats for user with email: {email}")

        if not email:
            logger.error("Email not provided")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Email is required"
            )

        # Get user from database
        user = await get_user_by_email(email)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found"
            )

        logger.info(f"Successfully retrieved stats for user with email: {email}")

        # Extract stats fields
        stats = {
            "workout_streak": user.get("workout_streak", 0),
            "total_workouts": user.get("total_workouts", 0),
            "active_minutes": user.get("active_minutes", 0),
            "calories_burned": user.get("calories_burned", 0),
            "badges": user.get("badges", []),
            "fitness_level": user.get("fitness_level", "beginner"),
        }

        return stats
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_user_stats: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving user stats: {str(e)}",
        )


@router.get("/leaderboard", response_model=List[Dict[str, Any]])
async def get_user_leaderboard(
    limit: int = Query(10, ge=1, le=100),
    email: Optional[str] = None,  # Optional email for user context
):
    """
    Get the top users by workout streak for the leaderboard
    """
    try:
        logger.info(f"Getting leaderboard with limit: {limit}")

        leaderboard = await get_leaderboard(limit)

        # Convert MongoDB _id to string for serialization in each user
        for user in leaderboard:
            if "_id" in user:
                user["_id"] = str(user["_id"])

        logger.info(f"Successfully retrieved leaderboard with {len(leaderboard)} users")

        return leaderboard
    except Exception as e:
        logger.error(f"Unexpected error in get_user_leaderboard: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving leaderboard: {str(e)}",
        )
