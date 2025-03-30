# api/exercise_routes.py
from fastapi import APIRouter, HTTPException, status, Query, Path, Body, Depends
from typing import Dict, Any, Optional, List
import logging
import sys
import traceback
import uuid
from datetime import datetime

from core.models.exercise import ExerciseCreate, ExerciseUpdate
from core.db_operations import (
    create_exercise,
    get_exercise_by_id,
    get_exercises_by_user_email,
    update_exercise,
    delete_exercise,
    get_user_exercise_stats,
)

# Enhanced logging
logger = logging.getLogger("morphos-exercise")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

router = APIRouter(prefix="/exercises", tags=["exercises"])


@router.post("", response_model=Dict[str, Any])
async def add_exercise(exercise_data: ExerciseCreate):
    """
    Create a new exercise session with exercise data
    """
    try:
        logger.info(
            f"Creating exercise for user with email: {exercise_data.user_email}"
        )

        # Convert to dictionary and add UUID
        exercise_dict = exercise_data.dict()
        exercise_dict["id"] = str(uuid.uuid4())

        # Create exercise in database
        new_exercise = await create_exercise(exercise_dict)

        if new_exercise is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error creating exercise session",
            )

        logger.info(
            f"Successfully created exercise for user: {exercise_data.user_email}"
        )

        # Convert MongoDB _id to string for serialization
        if "_id" in new_exercise:
            new_exercise["_id"] = str(new_exercise["_id"])

        return new_exercise
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in add_exercise: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating exercise session: {str(e)}",
        )


@router.get("/{exercise_id}", response_model=Dict[str, Any])
async def get_exercise(
    exercise_id: str = Path(..., description="The ID of the exercise to get")
):
    """
    Get a specific exercise session by ID
    """
    try:
        logger.info(f"Getting exercise with ID: {exercise_id}")

        # Get exercise from database
        exercise = await get_exercise_by_id(exercise_id)

        if exercise is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Exercise session not found",
            )

        logger.info(f"Successfully retrieved exercise with ID: {exercise_id}")

        # Convert MongoDB _id to string for serialization
        if "_id" in exercise:
            exercise["_id"] = str(exercise["_id"])

        return exercise
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_exercise: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving exercise: {str(e)}",
        )


@router.get("", response_model=List[Dict[str, Any]])
async def get_user_exercises(
    email: str,
    limit: int = Query(20, ge=1, le=100),
    skip: int = Query(0, ge=0),
):
    """
    Get exercise sessions for a specific user with pagination
    """
    try:
        logger.info(f"Getting exercises for user with email: {email}")

        if not email:
            logger.error("Email not provided")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Email is required"
            )

        # Get exercises from database
        exercises = await get_exercises_by_user_email(email, limit, skip)

        logger.info(
            f"Successfully retrieved {len(exercises)} exercises for user: {email}"
        )

        # Convert MongoDB _id to string for serialization in each exercise
        for exercise in exercises:
            if "_id" in exercise:
                exercise["_id"] = str(exercise["_id"])

        return exercises
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_user_exercises: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving user exercises: {str(e)}",
        )


@router.put("/{exercise_id}", response_model=Dict[str, Any])
async def update_exercise_session(
    exercise_id: str = Path(..., description="The ID of the exercise to update"),
    update_data: ExerciseUpdate = Body(...),
):
    """
    Update a specific exercise session
    """
    try:
        logger.info(f"Updating exercise with ID: {exercise_id}")

        # Convert to dictionary and exclude None values
        update_dict = update_data.dict(exclude_unset=True, exclude_none=True)

        if not update_dict:
            logger.warning("No valid update data provided")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid fields to update",
            )

        # Update exercise in database
        updated_exercise = await update_exercise(exercise_id, update_dict)

        if updated_exercise is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Exercise session not found",
            )

        logger.info(f"Successfully updated exercise with ID: {exercise_id}")

        # Convert MongoDB _id to string for serialization
        if "_id" in updated_exercise:
            updated_exercise["_id"] = str(updated_exercise["_id"])

        return updated_exercise
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in update_exercise_session: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating exercise session: {str(e)}",
        )


@router.delete("/{exercise_id}", response_model=Dict[str, str])
async def delete_exercise_session(
    exercise_id: str = Path(..., description="The ID of the exercise to delete")
):
    """
    Delete a specific exercise session
    """
    try:
        logger.info(f"Deleting exercise with ID: {exercise_id}")

        # Delete exercise from database
        result = await delete_exercise(exercise_id)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Exercise session not found",
            )

        logger.info(f"Successfully deleted exercise with ID: {exercise_id}")

        return {"message": "Exercise deleted successfully"}
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in delete_exercise_session: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting exercise session: {str(e)}",
        )


@router.get("/stats/{email}", response_model=Dict[str, Any])
async def get_exercise_stats(
    email: str = Path(..., description="The email of the user to get stats for")
):
    """
    Get exercise statistics for a specific user
    """
    try:
        logger.info(f"Getting exercise stats for user with email: {email}")

        # Get exercise stats from database
        stats = await get_user_exercise_stats(email)

        logger.info(f"Successfully retrieved exercise stats for user: {email}")

        return stats
    except Exception as e:
        logger.error(f"Unexpected error in get_exercise_stats: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving exercise stats: {str(e)}",
        )
