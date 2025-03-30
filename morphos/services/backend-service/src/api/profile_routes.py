from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Dict, Any, Optional, List
import logging
from core.security import get_current_user
from core.models.user import UserUpdate, UserProfile
from core.db_operations import (
    get_user_by_auth0_id,
    update_user_profile,
    update_user_achievements,
    get_leaderboard,
)

logger = logging.getLogger("morphos-profile")

router = APIRouter(prefix="/profile", tags=["profile"])


@router.get("/me", response_model=Dict[str, Any])
async def get_profile(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get the current user's profile with all fitness data
    """
    user_id = current_user.get("sub")  # Auth0 user ID is in the 'sub' claim

    # Get user from database
    user = await get_user_by_auth0_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found"
        )

    # Convert MongoDB _id to string for serialization
    if "_id" in user:
        user["_id"] = str(user["_id"])

    return user


@router.put("/me", response_model=Dict[str, Any])
async def update_profile(
    profile_data: UserUpdate, current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Update the current user's profile
    """
    user_id = current_user.get("sub")  # Auth0 user ID

    # Prepare update data - exclude None values
    update_data = profile_data.dict(exclude_unset=True, exclude_none=True)

    # Convert enum values to strings if present
    if "fitness_level" in update_data and update_data["fitness_level"]:
        update_data["fitness_level"] = update_data["fitness_level"].value

    # Update user in database
    updated_user = await update_user_profile(user_id, update_data)
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User profile not found or update failed",
        )

    # Convert MongoDB _id to string for serialization
    if "_id" in updated_user:
        updated_user["_id"] = str(updated_user["_id"])

    return updated_user


@router.post("/achievements", response_model=Dict[str, Any])
async def update_achievements(
    achievement_data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Update user achievement metrics (workout streak, total workouts, etc.)

    Can be used to increment counters, add badges, etc.
    """
    user_id = current_user.get("sub")  # Auth0 user ID

    # Update achievements in database
    updated_user = await update_user_achievements(user_id, achievement_data)
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User profile not found or update failed",
        )

    # Convert MongoDB _id to string for serialization
    if "_id" in updated_user:
        updated_user["_id"] = str(updated_user["_id"])

    return updated_user


@router.get("/leaderboard", response_model=List[Dict[str, Any]])
async def get_user_leaderboard(
    limit: int = Query(10, ge=1, le=100),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get the top users by workout streak for the leaderboard
    """
    leaderboard = await get_leaderboard(limit)

    # Convert MongoDB _id to string for serialization in each user
    for user in leaderboard:
        if "_id" in user:
            user["_id"] = str(user["_id"])

    return leaderboard


@router.get("/stats")
async def get_user_stats(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get the current user's workout statistics summary
    """
    user_id = current_user.get("sub")  # Auth0 user ID

    # Get user from database
    user = await get_user_by_auth0_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User profile not found"
        )

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
