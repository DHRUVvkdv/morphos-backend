# MongoDB User Schema

This document outlines the MongoDB schema for storing user profiles with fitness data.

## User Collection

The `users` collection stores all user information, including authentication details, physical stats, fitness information, and achievement tracking.

### Fields

#### User Identity

- `_id`: ObjectId (MongoDB generated ID)
- `auth0_id`: String (Auth0 User ID for authentication)
- `email`: String (User's email address, unique)
- `name`: String (User's full name)
- `profile_picture`: String (URL to profile picture, optional)

#### Physical Stats

- `height`: Number (User's height in cm)
- `weight`: Number (User's weight in kg)
- `age`: Number (User's age in years)
- `bmi`: Number (Calculated Body Mass Index)

#### Fitness Information

- `fitness_level`: String (One of: "beginner", "intermediate", "advanced")
- `workout_duration`: Number (User's preferred workout duration in minutes, default: 45)
- `workout_frequency`: Number (User's preferred workouts per week, default: 4)
- `fitness_goals`: Array of Strings (e.g., ["build_muscle", "lose_weight"])
- `available_equipment`: Array of Strings (e.g., ["dumbbells", "resistance_bands"])

#### Achievement Tracking

- `workout_streak`: Number (Current streak of consecutive workout days)
- `total_workouts`: Number (Total workouts completed)
- `active_minutes`: Number (Total active minutes)
- `calories_burned`: Number (Total calories burned)
- `badges`: Array of Strings (Achievement badges earned)

#### System Fields

- `created_at`: Date (Account creation timestamp)
- `updated_at`: Date (Last update timestamp)

### Indexes

- `email`: Unique index for efficient user lookup by email
- `auth0_id`: Unique index for efficient user lookup by Auth0 ID

## Example Document

```json
{
  "_id": ObjectId("507f1f77bcf86cd799439011"),
  "auth0_id": "auth0|123456789",
  "email": "user@example.com",
  "name": "John Doe",
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
  "workout_streak": 5,
  "total_workouts": 23,
  "active_minutes": 1200,
  "calories_burned": 15000,
  "badges": ["first_workout", "week_streak"],
  "created_at": ISODate("2023-03-15T12:00:00Z"),
  "updated_at": ISODate("2023-03-20T15:30:00Z")
}
```

# MongoDB Exercises Schema

This document outlines the MongoDB schema for storing exercise sessions with performance data.

## Exercises Collection

The `exercises` collection stores all workout session information, including exercise data and performance metrics.

### Fields

#### Session Identification

- `_id`: ObjectId (MongoDB generated ID)
- `id`: String (UUID, unique identifier for the exercise session)
- `user_email`: String (Email of the user who performed the workout)

#### Session Timing

- `date`: Date (Date the workout was completed)
- `start_time`: DateTime (When the workout started)
- `end_time`: DateTime (When the workout ended)
- `duration_minutes`: Integer (Total duration in minutes)

#### T-Pose Exercise

- `tpose_performed`: Boolean (Whether this exercise was performed)
- `tpose_hold_time_seconds`: Integer (Hold time for T-Pose)
- `tpose_form_score`: Float (Form score for T-Pose, between 0 and 1)

#### Bicep Curl Exercise

- `bicep_curl_performed`: Boolean (Whether this exercise was performed)
- `bicep_curl_reps`: Integer (Reps for Bicep Curl)
- `bicep_curl_form_score`: Float (Form score for Bicep Curl, between 0 and 1)

#### Squat Exercise

- `squat_performed`: Boolean (Whether this exercise was performed)
- `squat_reps`: Integer (Reps for Squat)
- `squat_form_score`: Float (Form score for Squat, between 0 and 1)

#### Lateral Raise Exercise

- `lateral_raise_performed`: Boolean (Whether this exercise was performed)
- `lateral_raise_reps`: Integer (Reps for Lateral Raise)
- `lateral_raise_form_score`: Float (Form score for Lateral Raise, between 0 and 1)

#### Plank Exercise

- `plank_performed`: Boolean (Whether this exercise was performed)
- `plank_hold_time_seconds`: Integer (Hold time for Plank)
- `plank_form_score`: Float (Form score for Plank, between 0 and 1)

#### System Fields

- `created_at`: DateTime (Record creation timestamp)

### Indexes

- `user_email`: Index for efficient lookup by user
- `id`: Unique index for efficient lookup by exercise ID

## Example Document

```json
{
  "_id": ObjectId("507f1f77bcf86cd799439011"),
  "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "user_email": "user@example.com",
  "date": ISODate("2023-03-15T00:00:00Z"),
  "start_time": ISODate("2023-03-15T12:00:00Z"),
  "end_time": ISODate("2023-03-15T12:45:00Z"),
  "duration_minutes": 45,
  "tpose_performed": true,
  "tpose_hold_time_seconds": 30,
  "tpose_form_score": 0.92,
  "bicep_curl_performed": true,
  "bicep_curl_reps": 12,
  "bicep_curl_form_score": 0.85,
  "squat_performed": true,
  "squat_reps": 15,
  "squat_form_score": 0.78,
  "lateral_raise_performed": false,
  "lateral_raise_reps": null,
  "lateral_raise_form_score": null,
  "plank_performed": true,
  "plank_hold_time_seconds": 60,
  "plank_form_score": 0.88,
  "created_at": ISODate("2023-03-15T12:45:30Z")
}
```
