from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError
from bson.objectid import ObjectId
from core.database import get_db

logger = logging.getLogger("morphos-db-ops")


# User Collection Operations
async def create_user(user_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Create a new user in the database

    Args:
        user_data: User data including auth0_id, email, name, etc.

    Returns:
        The created user document or None if error
    """
    db = get_db()
    if db is None:  # Changed from "if not db:" to "if db is None:"
        logger.error("Database connection not available")
        return None

    # Ensure required fields
    if not user_data.get("auth0_id") or not user_data.get("email"):
        logger.error("Missing required fields for user creation")
        return None

    # Add default values for achievement tracking fields
    user_data.setdefault("workout_streak", 0)
    user_data.setdefault("total_workouts", 0)
    user_data.setdefault("active_minutes", 0)
    user_data.setdefault("calories_burned", 0)
    user_data.setdefault("badges", [])

    # Set default workout preferences if not provided
    user_data.setdefault("workout_duration", 45)  # Default 45 minutes
    user_data.setdefault("workout_frequency", 4)  # Default 4 times per week

    # Initialize empty lists if not provided
    user_data.setdefault("fitness_goals", [])
    user_data.setdefault("available_equipment", [])

    # Calculate BMI if height and weight are provided
    if user_data.get("height") and user_data.get("weight"):
        height_m = user_data["height"] / 100  # Convert cm to meters
        user_data["bmi"] = user_data["weight"] / (height_m * height_m)

    # Add timestamps
    now = datetime.utcnow()
    user_data["created_at"] = now
    user_data["updated_at"] = now

    try:
        result = db.users.insert_one(user_data)
        if result.inserted_id:
            # Fetch the inserted document
            return db.users.find_one({"_id": result.inserted_id})
        return None
    except DuplicateKeyError:
        logger.error(f"User with email {user_data.get('email')} already exists")
        return None
    except PyMongoError as e:
        logger.error(f"Error creating user: {str(e)}")
        return None


async def get_user_by_auth0_id(auth0_id: str) -> Optional[Dict[str, Any]]:
    """
    Get user by Auth0 ID

    Args:
        auth0_id: Auth0 user ID

    Returns:
        User document or None if not found
    """
    db = get_db()
    if db is None:  # Changed from "if not db:" to "if db is None:"
        return None

    try:
        return db.users.find_one({"auth0_id": auth0_id})
    except PyMongoError as e:
        logger.error(f"Error fetching user by auth0_id: {str(e)}")
        return None


async def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """
    Get user by email

    Args:
        email: User email

    Returns:
        User document or None if not found
    """
    db = get_db()
    if db is None:  # Changed from "if not db:" to "if db is None:"
        return None

    try:
        return db.users.find_one({"email": email})
    except PyMongoError as e:
        logger.error(f"Error fetching user by email: {str(e)}")
        return None


async def update_user_profile(
    auth0_id: str, update_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Update user profile

    Args:
        auth0_id: Auth0 user ID
        update_data: Data to update

    Returns:
        Updated user document or None if error
    """
    db = get_db()
    if not db:
        return None

    # Add updated_at timestamp
    update_data["updated_at"] = datetime.utcnow()

    # Calculate BMI if height and weight are being updated
    height = update_data.get("height")
    weight = update_data.get("weight")

    if height is not None or weight is not None:
        # Get current user data to see if we have both height and weight
        current_user = await get_user_by_auth0_id(auth0_id)
        if current_user:
            # If height is being updated, use that; otherwise use existing height
            actual_height = height if height is not None else current_user.get("height")
            # If weight is being updated, use that; otherwise use existing weight
            actual_weight = weight if weight is not None else current_user.get("weight")

            # Calculate BMI if we have both values
            if actual_height and actual_weight:
                height_m = actual_height / 100  # Convert cm to meters
                update_data["bmi"] = actual_weight / (height_m * height_m)

    try:
        # Find and update in one operation, returning the updated document
        updated_user = db.users.find_one_and_update(
            {"auth0_id": auth0_id},
            {"$set": update_data},
            return_document=ReturnDocument.AFTER,
        )
        return updated_user
    except PyMongoError as e:
        logger.error(f"Error updating user profile: {str(e)}")
        return None


async def update_user_achievements(
    auth0_id: str, achievement_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Update user achievement metrics (workout streak, total workouts, etc.)

    Args:
        auth0_id: Auth0 user ID
        achievement_data: Achievement data to update

    Returns:
        Updated user document or None if error
    """
    db = get_db()
    if not db:
        return None

    valid_achievement_fields = {
        "workout_streak",
        "total_workouts",
        "active_minutes",
        "calories_burned",
        "badges",
    }

    # Filter to only include valid achievement fields
    filtered_data = {
        k: v for k, v in achievement_data.items() if k in valid_achievement_fields
    }

    if not filtered_data:
        logger.warning("No valid achievement fields to update")
        return None

    # Add updated_at timestamp
    filtered_data["updated_at"] = datetime.utcnow()

    try:
        # For numeric fields that should be incremented, not replaced
        increment_fields = {}
        if "active_minutes" in filtered_data:
            increment_fields["active_minutes"] = filtered_data.pop("active_minutes")
        if "calories_burned" in filtered_data:
            increment_fields["calories_burned"] = filtered_data.pop("calories_burned")
        if "total_workouts" in filtered_data:
            increment_fields["total_workouts"] = filtered_data.pop("total_workouts")

        # Add badges (if any) to the badges array without duplicates
        add_to_set = {}
        if "badges" in filtered_data:
            add_to_set["badges"] = {"$each": filtered_data.pop("badges")}

        # Build the update operation
        update_op = {}
        if filtered_data:
            update_op["$set"] = filtered_data
        if increment_fields:
            update_op["$inc"] = increment_fields
        if add_to_set:
            update_op["$addToSet"] = add_to_set

        # Find and update in one operation, returning the updated document
        updated_user = db.users.find_one_and_update(
            {"auth0_id": auth0_id}, update_op, return_document=ReturnDocument.AFTER
        )
        return updated_user
    except PyMongoError as e:
        logger.error(f"Error updating user achievements: {str(e)}")
        return None


async def delete_user(auth0_id: str) -> bool:
    """
    Delete a user

    Args:
        auth0_id: Auth0 user ID

    Returns:
        True if deleted successfully, False otherwise
    """
    db = get_db()
    if not db:
        return False

    try:
        result = db.users.delete_one({"auth0_id": auth0_id})
        return result.deleted_count > 0
    except PyMongoError as e:
        logger.error(f"Error deleting user: {str(e)}")
        return False


async def get_leaderboard(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get top users by workout streak for leaderboard

    Args:
        limit: Maximum number of users to return

    Returns:
        List of user documents sorted by workout streak
    """
    db = get_db()
    if db is None:  # Changed from "if not db:" to "if db is None:"
        logger.error("Database connection not available")
        return []

    try:
        # Project only fields needed for leaderboard
        projection = {
            "name": 1,
            "profile_picture": 1,
            "workout_streak": 1,
            "total_workouts": 1,
            "badges": 1,
        }

        # Sort by workout streak descending
        cursor = db.users.find({}, projection).sort("workout_streak", -1).limit(limit)

        return [user for user in cursor]
    except PyMongoError as e:
        logger.error(f"Error fetching leaderboard: {str(e)}")
        return []


async def update_user_profile_by_email(
    email: str, update_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Update user profile by email

    Args:
        email: User email
        update_data: Data to update

    Returns:
        Updated user document or None if error
    """
    db = get_db()
    if db is None:
        logger.error("Database connection not available")
        return None

    # Add updated_at timestamp
    update_data["updated_at"] = datetime.utcnow()

    # Calculate BMI if height and weight are being updated
    height = update_data.get("height")
    weight = update_data.get("weight")

    if height is not None or weight is not None:
        # Get current user data to see if we have both height and weight
        current_user = await get_user_by_email(email)
        if current_user is not None:
            # If height is being updated, use that; otherwise use existing height
            actual_height = height if height is not None else current_user.get("height")
            # If weight is being updated, use that; otherwise use existing weight
            actual_weight = weight if weight is not None else current_user.get("weight")

            # Calculate BMI if we have both values
            if actual_height and actual_weight:
                height_m = actual_height / 100  # Convert cm to meters
                update_data["bmi"] = actual_weight / (height_m * height_m)

    try:
        # Find and update in one operation, returning the updated document
        updated_user = db.users.find_one_and_update(
            {"email": email},
            {"$set": update_data},
            return_document=ReturnDocument.AFTER,
        )
        logger.info(f"Updated profile for user with email: {email}")
        return updated_user
    except PyMongoError as e:
        logger.error(f"Error updating user profile by email: {str(e)}")
        return None


async def update_user_achievements_by_email(
    email: str, achievement_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Update user achievement metrics (workout streak, total workouts, etc.) by email

    Args:
        email: User email
        achievement_data: Achievement data to update

    Returns:
        Updated user document or None if error
    """
    db = get_db()
    if db is None:
        logger.error("Database connection not available")
        return None

    valid_achievement_fields = {
        "workout_streak",
        "total_workouts",
        "active_minutes",
        "calories_burned",
        "badges",
    }

    # Filter to only include valid achievement fields
    filtered_data = {
        k: v for k, v in achievement_data.items() if k in valid_achievement_fields
    }

    if not filtered_data:
        logger.warning("No valid achievement fields to update")
        return None

    # Add updated_at timestamp
    filtered_data["updated_at"] = datetime.utcnow()

    try:
        # For numeric fields that should be incremented, not replaced
        increment_fields = {}
        if "active_minutes" in filtered_data:
            increment_fields["active_minutes"] = filtered_data.pop("active_minutes")
        if "calories_burned" in filtered_data:
            increment_fields["calories_burned"] = filtered_data.pop("calories_burned")
        if "total_workouts" in filtered_data:
            increment_fields["total_workouts"] = filtered_data.pop("total_workouts")

        # Add badges (if any) to the badges array without duplicates
        add_to_set = {}
        if "badges" in filtered_data:
            add_to_set["badges"] = {"$each": filtered_data.pop("badges")}

        # Build the update operation
        update_op = {}
        if filtered_data:
            update_op["$set"] = filtered_data
        if increment_fields:
            update_op["$inc"] = increment_fields
        if add_to_set:
            update_op["$addToSet"] = add_to_set

        # Find and update in one operation, returning the updated document
        updated_user = db.users.find_one_and_update(
            {"email": email}, update_op, return_document=ReturnDocument.AFTER
        )
        logger.info(f"Updated achievements for user with email: {email}")
        return updated_user
    except PyMongoError as e:
        logger.error(f"Error updating user achievements by email: {str(e)}")
        return None


async def delete_user_by_email(email: str) -> bool:
    """
    Delete a user by email

    Args:
        email: User email

    Returns:
        True if deleted successfully, False otherwise
    """
    db = get_db()
    if db is None:
        logger.error("Database connection not available")
        return False

    try:
        result = db.users.delete_one({"email": email})
        success = result.deleted_count > 0
        if success:
            logger.info(f"Deleted user with email: {email}")
        else:
            logger.warning(f"User with email {email} not found for deletion")
        return success
    except PyMongoError as e:
        logger.error(f"Error deleting user by email: {str(e)}")
        return False


# Exercise Collection Operations
async def create_exercise(exercise_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Create a new exercise session in the database
    Args:
        exercise_data: Exercise session data including user_email, exercises performed, etc.
    Returns:
        The created exercise document or None if error
    """
    db = get_db()
    if db is None:
        logger.error("Database connection not available")
        return None

    # Ensure required fields
    if not exercise_data.get("user_email"):
        logger.error("Missing required field 'user_email' for exercise creation")
        return None

    # Add a UUID if not provided
    if not exercise_data.get("id"):
        exercise_data["id"] = str(uuid.uuid4())

    # Add timestamps
    now = datetime.utcnow()
    exercise_data["created_at"] = now

    try:
        result = db.exercises.insert_one(exercise_data)
        if result.inserted_id:
            # Fetch the inserted document
            return db.exercises.find_one({"_id": result.inserted_id})
        return None
    except PyMongoError as e:
        logger.error(f"Error creating exercise: {str(e)}")
        return None


async def get_exercise_by_id(exercise_id: str) -> Optional[Dict[str, Any]]:
    """
    Get exercise by ID
    Args:
        exercise_id: Exercise session ID
    Returns:
        Exercise document or None if not found
    """
    db = get_db()
    if db is None:
        return None
    try:
        return db.exercises.find_one({"id": exercise_id})
    except PyMongoError as e:
        logger.error(f"Error fetching exercise by id: {str(e)}")
        return None


async def get_exercises_by_user_email(
    email: str, limit: int = 20, skip: int = 0
) -> List[Dict[str, Any]]:
    """
    Get exercises by user email with pagination
    Args:
        email: User email
        limit: Maximum number of exercises to return
        skip: Number of exercises to skip (for pagination)
    Returns:
        List of exercise documents
    """
    db = get_db()
    if db is None:
        return []
    try:
        cursor = (
            db.exercises.find({"user_email": email})
            .sort("date", -1)
            .skip(skip)
            .limit(limit)
        )
        return [exercise for exercise in cursor]
    except PyMongoError as e:
        logger.error(f"Error fetching exercises by email: {str(e)}")
        return []


async def update_exercise(
    exercise_id: str, update_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Update exercise session
    Args:
        exercise_id: Exercise session ID
        update_data: Data to update
    Returns:
        Updated exercise document or None if error
    """
    db = get_db()
    if db is None:
        return None

    try:
        # Find and update in one operation, returning the updated document
        updated_exercise = db.exercises.find_one_and_update(
            {"id": exercise_id},
            {"$set": update_data},
            return_document=ReturnDocument.AFTER,
        )
        return updated_exercise
    except PyMongoError as e:
        logger.error(f"Error updating exercise: {str(e)}")
        return None


async def delete_exercise(exercise_id: str) -> bool:
    """
    Delete an exercise session
    Args:
        exercise_id: Exercise session ID
    Returns:
        True if deleted successfully, False otherwise
    """
    db = get_db()
    if db is None:
        return False
    try:
        result = db.exercises.delete_one({"id": exercise_id})
        return result.deleted_count > 0
    except PyMongoError as e:
        logger.error(f"Error deleting exercise: {str(e)}")
        return False


async def get_user_exercise_stats(email: str) -> Dict[str, Any]:
    """
    Get exercise statistics for a user
    Args:
        email: User email
    Returns:
        Dictionary with exercise statistics
    """
    db = get_db()
    if db is None:
        return {
            "total_exercises": 0,
            "total_duration_minutes": 0,
            "average_duration_minutes": 0,
            "exercises_performed": {},
        }

    try:
        # Count total exercises - fixed to not use await
        total_exercises = db.exercises.count_documents({"user_email": email})

        # If no exercises, return empty stats
        if total_exercises == 0:
            return {
                "total_exercises": 0,
                "total_duration_minutes": 0,
                "average_duration_minutes": 0,
                "exercises_performed": {},
            }

        # Aggregate exercise stats
        pipeline = [
            {"$match": {"user_email": email}},
            {
                "$group": {
                    "_id": None,
                    "total_duration": {"$sum": "$duration_minutes"},
                    "avg_duration": {"$avg": "$duration_minutes"},
                    "tpose_count": {
                        "$sum": {"$cond": [{"$eq": ["$tpose_performed", True]}, 1, 0]}
                    },
                    "bicep_curl_count": {
                        "$sum": {
                            "$cond": [{"$eq": ["$bicep_curl_performed", True]}, 1, 0]
                        }
                    },
                    "squat_count": {
                        "$sum": {"$cond": [{"$eq": ["$squat_performed", True]}, 1, 0]}
                    },
                    "lateral_raise_count": {
                        "$sum": {
                            "$cond": [{"$eq": ["$lateral_raise_performed", True]}, 1, 0]
                        }
                    },
                    "plank_count": {
                        "$sum": {"$cond": [{"$eq": ["$plank_performed", True]}, 1, 0]}
                    },
                    "avg_tpose_score": {
                        "$avg": {
                            "$cond": [
                                {"$eq": ["$tpose_performed", True]},
                                "$tpose_form_score",
                                None,
                            ]
                        }
                    },
                    "avg_bicep_curl_score": {
                        "$avg": {
                            "$cond": [
                                {"$eq": ["$bicep_curl_performed", True]},
                                "$bicep_curl_form_score",
                                None,
                            ]
                        }
                    },
                    "avg_squat_score": {
                        "$avg": {
                            "$cond": [
                                {"$eq": ["$squat_performed", True]},
                                "$squat_form_score",
                                None,
                            ]
                        }
                    },
                    "avg_lateral_raise_score": {
                        "$avg": {
                            "$cond": [
                                {"$eq": ["$lateral_raise_performed", True]},
                                "$lateral_raise_form_score",
                                None,
                            ]
                        }
                    },
                    "avg_plank_score": {
                        "$avg": {
                            "$cond": [
                                {"$eq": ["$plank_performed", True]},
                                "$plank_form_score",
                                None,
                            ]
                        }
                    },
                }
            },
        ]

        # Fixed to not use await with to_list
        result = list(db.exercises.aggregate(pipeline))

        if not result:
            return {
                "total_exercises": total_exercises,
                "total_duration_minutes": 0,
                "average_duration_minutes": 0,
                "exercises_performed": {},
            }

        stats = result[0]

        return {
            "total_exercises": total_exercises,
            "total_duration_minutes": stats.get("total_duration", 0),
            "average_duration_minutes": round(stats.get("avg_duration", 0), 1),
            "exercises_performed": {
                "tpose": {
                    "count": stats.get("tpose_count", 0),
                    "avg_form_score": round(stats.get("avg_tpose_score", 0) or 0, 2),
                },
                "bicep_curl": {
                    "count": stats.get("bicep_curl_count", 0),
                    "avg_form_score": round(
                        stats.get("avg_bicep_curl_score", 0) or 0, 2
                    ),
                },
                "squat": {
                    "count": stats.get("squat_count", 0),
                    "avg_form_score": round(stats.get("avg_squat_score", 0) or 0, 2),
                },
                "lateral_raise": {
                    "count": stats.get("lateral_raise_count", 0),
                    "avg_form_score": round(
                        stats.get("avg_lateral_raise_score", 0) or 0, 2
                    ),
                },
                "plank": {
                    "count": stats.get("plank_count", 0),
                    "avg_form_score": round(stats.get("avg_plank_score", 0) or 0, 2),
                },
            },
        }
    except PyMongoError as e:
        logger.error(f"Error getting exercise stats: {str(e)}")
        return {
            "total_exercises": 0,
            "total_duration_minutes": 0,
            "average_duration_minutes": 0,
            "exercises_performed": {},
        }
