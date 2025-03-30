# core/models/exercise.py
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from datetime import datetime
import uuid


class Exercise(BaseModel):
    """Exercise session data model with exercise data"""

    # Identification
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_email: EmailStr

    # Session timing
    date: datetime
    start_time: datetime
    end_time: datetime
    duration_minutes: int

    # T-Pose
    tpose_performed: bool = False
    tpose_hold_time_seconds: Optional[int] = None
    tpose_form_score: Optional[float] = None

    # Bicep Curl
    bicep_curl_performed: bool = False
    bicep_curl_reps: Optional[int] = None
    bicep_curl_form_score: Optional[float] = None

    # Squat
    squat_performed: bool = False
    squat_reps: Optional[int] = None
    squat_form_score: Optional[float] = None

    # Lateral Raise
    lateral_raise_performed: bool = False
    lateral_raise_reps: Optional[int] = None
    lateral_raise_form_score: Optional[float] = None

    # Plank
    plank_performed: bool = False
    plank_hold_time_seconds: Optional[int] = None
    plank_form_score: Optional[float] = None

    # System fields
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                "user_email": "user@example.com",
                "date": "2023-03-15T12:00:00Z",
                "start_time": "2023-03-15T12:00:00Z",
                "end_time": "2023-03-15T12:45:00Z",
                "duration_minutes": 45,
                "tpose_performed": True,
                "tpose_hold_time_seconds": 30,
                "tpose_form_score": 0.92,
                "bicep_curl_performed": True,
                "bicep_curl_reps": 12,
                "bicep_curl_form_score": 0.85,
                "squat_performed": True,
                "squat_reps": 15,
                "squat_form_score": 0.78,
                "lateral_raise_performed": False,
                "plank_performed": True,
                "plank_hold_time_seconds": 60,
                "plank_form_score": 0.88,
                "created_at": "2023-03-15T12:45:30Z",
            }
        }


class ExerciseCreate(BaseModel):
    """Schema for creating exercise sessions"""

    user_email: EmailStr
    date: datetime
    start_time: datetime
    end_time: datetime
    duration_minutes: int

    # T-Pose
    tpose_performed: bool = False
    tpose_hold_time_seconds: Optional[int] = None
    tpose_form_score: Optional[float] = None

    # Bicep Curl
    bicep_curl_performed: bool = False
    bicep_curl_reps: Optional[int] = None
    bicep_curl_form_score: Optional[float] = None

    # Squat
    squat_performed: bool = False
    squat_reps: Optional[int] = None
    squat_form_score: Optional[float] = None

    # Lateral Raise
    lateral_raise_performed: bool = False
    lateral_raise_reps: Optional[int] = None
    lateral_raise_form_score: Optional[float] = None

    # Plank
    plank_performed: bool = False
    plank_hold_time_seconds: Optional[int] = None
    plank_form_score: Optional[float] = None


class ExerciseUpdate(BaseModel):
    """Schema for updating exercise sessions"""

    date: Optional[datetime] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_minutes: Optional[int] = None

    # T-Pose
    tpose_performed: Optional[bool] = None
    tpose_hold_time_seconds: Optional[int] = None
    tpose_form_score: Optional[float] = None

    # Bicep Curl
    bicep_curl_performed: Optional[bool] = None
    bicep_curl_reps: Optional[int] = None
    bicep_curl_form_score: Optional[float] = None

    # Squat
    squat_performed: Optional[bool] = None
    squat_reps: Optional[int] = None
    squat_form_score: Optional[float] = None

    # Lateral Raise
    lateral_raise_performed: Optional[bool] = None
    lateral_raise_reps: Optional[int] = None
    lateral_raise_form_score: Optional[float] = None

    # Plank
    plank_performed: Optional[bool] = None
    plank_hold_time_seconds: Optional[int] = None
    plank_form_score: Optional[float] = None
