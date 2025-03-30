from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List, Dict, Any, Set
from datetime import datetime
from enum import Enum


class FitnessLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class FitnessGoals(str, Enum):
    BUILD_MUSCLE = "build_muscle"
    LOSE_WEIGHT = "lose_weight"
    IMPROVE_FLEXIBILITY = "improve_flexibility"
    INCREASE_ENDURANCE = "increase_endurance"
    GENERAL_FITNESS = "general_fitness"


class Equipment(str, Enum):
    NONE = "none"
    DUMBBELLS = "dumbbells"
    RESISTANCE_BANDS = "resistance_bands"
    BARBELL = "barbell"
    KETTLEBELL = "kettlebell"
    PULL_UP_BAR = "pull_up_bar"
    EXERCISE_BIKE = "exercise_bike"
    TREADMILL = "treadmill"


class UserProfile(BaseModel):
    """User profile data model with all fields"""

    # User Identity
    name: str
    email: EmailStr
    auth0_id: str
    profile_picture: Optional[str] = None

    # Physical Stats
    height: Optional[float] = None  # in cm
    weight: Optional[float] = None  # in kg
    age: Optional[int] = None
    bmi: Optional[float] = None  # Calculated field

    # Fitness Information
    fitness_level: Optional[FitnessLevel] = None
    workout_duration: Optional[int] = Field(default=45)  # in minutes
    workout_frequency: Optional[int] = Field(default=4)  # workouts per week
    fitness_goals: List[str] = Field(default_factory=list)
    available_equipment: List[str] = Field(default_factory=list)

    # Achievement Tracking
    workout_streak: int = Field(default=0)
    total_workouts: int = Field(default=0)
    active_minutes: int = Field(default=0)
    calories_burned: int = Field(default=0)
    badges: List[str] = Field(default_factory=list)

    # System fields
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Validators
    @validator("height")
    def height_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Height must be positive")
        return v

    @validator("weight")
    def weight_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Weight must be positive")
        return v

    @validator("age")
    def age_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Age must be positive")
        return v

    @validator("workout_duration")
    def workout_duration_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Workout duration must be positive")
        return v

    @validator("workout_frequency")
    def workout_frequency_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Workout frequency must be positive")
        return v

    @validator("bmi", always=True)
    def calculate_bmi(cls, v, values):
        height = values.get("height")
        weight = values.get("weight")
        if height and weight:
            # Convert height from cm to meters
            height_m = height / 100
            # Calculate BMI: weight(kg) / (height(m))²
            return weight / (height_m * height_m)
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "name": "John Doe",
                "email": "john@example.com",
                "auth0_id": "auth0|123456789",
                "profile_picture": "https://example.com/profile.jpg",
                "height": 180,
                "weight": 75,
                "age": 30,
                "bmi": 23.15,
                "fitness_level": "intermediate",
                "workout_duration": 45,
                "workout_frequency": 4,
                "fitness_goals": ["build_muscle", "increase_endurance"],
                "available_equipment": ["dumbbells", "resistance_bands"],
                "workout_streak": 0,
                "total_workouts": 0,
                "active_minutes": 0,
                "calories_burned": 0,
                "badges": [],
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00",
            }
        }


class UserLogin(BaseModel):
    """Schema for user login"""

    email: EmailStr
    password: str


class UserCreate(BaseModel):
    """Schema for user creation during signup"""

    # Required fields
    email: EmailStr
    password: str
    name: str

    # Optional physical stats
    height: Optional[float] = None  # in cm
    weight: Optional[float] = None  # in kg
    age: Optional[int] = None

    # Optional fitness information
    fitness_level: Optional[FitnessLevel] = None
    workout_duration: Optional[int] = 45  # in minutes
    workout_frequency: Optional[int] = 4  # workouts per week
    fitness_goals: Optional[List[str]] = None
    available_equipment: Optional[List[str]] = None

    # Validators
    @validator("height")
    def height_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Height must be positive")
        return v

    @validator("weight")
    def weight_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Weight must be positive")
        return v

    @validator("age")
    def age_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Age must be positive")
        return v

    @validator("workout_duration")
    def workout_duration_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Workout duration must be positive")
        return v

    @validator("workout_frequency")
    def workout_frequency_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Workout frequency must be positive")
        return v


class UserUpdate(BaseModel):
    """Schema for user profile updates"""

    # User Identity
    name: Optional[str] = None
    profile_picture: Optional[str] = None

    # Physical Stats
    height: Optional[float] = None  # in cm
    weight: Optional[float] = None  # in kg
    age: Optional[int] = None

    # Fitness Information
    fitness_level: Optional[FitnessLevel] = None
    workout_duration: Optional[int] = None  # in minutes
    workout_frequency: Optional[int] = None  # workouts per week
    fitness_goals: Optional[List[str]] = None
    available_equipment: Optional[List[str]] = None

    # Validators
    @validator("height")
    def height_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Height must be positive")
        return v

    @validator("weight")
    def weight_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Weight must be positive")
        return v

    @validator("age")
    def age_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Age must be positive")
        return v

    @validator("workout_duration")
    def workout_duration_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Workout duration must be positive")
        return v

    @validator("workout_frequency")
    def workout_frequency_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Workout frequency must be positive")
        return v
