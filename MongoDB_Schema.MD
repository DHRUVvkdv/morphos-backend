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
