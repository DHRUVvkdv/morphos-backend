# Morphos API Documentation

This document provides details on the Morphos backend API endpoints, including authentication, user profile management, and fitness tracking features.

## Base URL

For local development: `http://localhost:8080`
For production: `https://morphos-backend-service-1020595365432.us-central1.run.app`

## Authentication Endpoints

### Sign Up

Register a new user account with fitness data.

- **URL**: `/auth/signup`
- **Method**: `POST`
- **Content-Type**: `application/json`

**Request Body**:

```json
{
  "email": "user@example.com",
  "password": "SecurePassword123",
  "name": "John Doe",
  "height": 180,
  "weight": 75,
  "age": 30,
  "fitness_level": "intermediate",
  "workout_duration": 45,
  "workout_frequency": 4,
  "fitness_goals": ["build_muscle", "increase_endurance"],
  "available_equipment": ["dumbbells", "resistance_bands"]
}
```

**Response (200 OK)**:

```json
{
  "status": "success",
  "message": "User created successfully",
  "user_id": "auth0|123456789"
}
```

**Error Responses**:

- `409 Conflict`: User with this email already exists
- `400 Bad Request`: Invalid data provided
- `500 Internal Server Error`: Server error

### Sign In

Authenticate and get an access token.

- **URL**: `/auth/signin`
- **Method**: `POST`
- **Content-Type**: `application/json`

**Request Body**:

```json
{
  "email": "user@example.com",
  "password": "SecurePassword123"
}
```

**Response (200 OK)**:

```json
{
  "access_token": "eyJhbGciO...",
  "token_type": "Bearer",
  "expires_in": 86400,
  "user": {
    "_id": "507f1f77bcf86cd799439011",
    "email": "user@example.com",
    "name": "John Doe",
    "fitness_level": "intermediate"
    // Other user data...
  }
}
```

**Error Responses**:

- `401 Unauthorized`: Invalid credentials
- `500 Internal Server Error`: Server error

## Profile Endpoints

All profile endpoints require authentication via Bearer token in the Authorization header.

### Get User Profile

Get the current user's complete profile.

- **URL**: `/profile/me`
- **Method**: `GET`
- **Authorization**: Bearer Token

**Response (200 OK)**:

```json
{
  "_id": "507f1f77bcf86cd799439011",
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
  "created_at": "2023-03-15T12:00:00Z",
  "updated_at": "2023-03-20T15:30:00Z"
}
```

**Error Responses**:

- `401 Unauthorized`: Not authenticated
- `404 Not Found`: User profile not found

### Update User Profile

Update the current user's profile data.

- **URL**: `/profile/me`
- **Method**: `PUT`
- **Authorization**: Bearer Token
- **Content-Type**: `application/json`

**Request Body** (all fields optional):

```json
{
  "name": "John Smith",
  "height": 182,
  "weight": 78,
  "age": 31,
  "fitness_level": "advanced",
  "workout_duration": 60,
  "workout_frequency": 5,
  "fitness_goals": [
    "build_muscle",
    "increase_endurance",
    "improve_flexibility"
  ],
  "available_equipment": ["dumbbells", "barbell", "pull_up_bar"]
}
```

**Response (200 OK)**:

```json
{
  "_id": "507f1f77bcf86cd799439011",
  "name": "John Smith",
  "email": "user@example.com"
  // Updated user data...
}
```

**Error Responses**:

- `401 Unauthorized`: Not authenticated
- `404 Not Found`: User profile not found
- `400 Bad Request`: Invalid data provided

### Update User Achievements

Update achievement metrics like workout streak, total workouts, etc.

- **URL**: `/profile/achievements`
- **Method**: `POST`
- **Authorization**: Bearer Token
- **Content-Type**: `application/json`

**Request Body**:

```json
{
  "workout_streak": 1,
  "total_workouts": 1,
  "active_minutes": 45,
  "calories_burned": 300,
  "badges": ["first_workout"]
}
```

**Response (200 OK)**:

```json
{
  "_id": "507f1f77bcf86cd799439011",
  "name": "John Smith",
  "workout_streak": 6,
  "total_workouts": 24,
  "active_minutes": 1245,
  "calories_burned": 15300,
  "badges": ["first_workout", "week_streak"]
  // Other user data...
}
```

**Error Responses**:

- `401 Unauthorized`: Not authenticated
- `404 Not Found`: User profile not found
- `400 Bad Request`: Invalid data provided

### Get User Stats

Get a summary of the user's fitness stats.

- **URL**: `/profile/stats`
- **Method**: `GET`
- **Authorization**: Bearer Token

**Response (200 OK)**:

```json
{
  "workout_streak": 6,
  "total_workouts": 24,
  "active_minutes": 1245,
  "calories_burned": 15300,
  "badges": ["first_workout", "week_streak"],
  "fitness_level": "intermediate"
}
```

**Error Responses**:

- `401 Unauthorized`: Not authenticated
- `404 Not Found`: User profile not found

### Get Leaderboard

Get the top users by workout streak.

- **URL**: `/profile/leaderboard?limit=10`
- **Method**: `GET`
- **Authorization**: Bearer Token
- **Query Parameters**:
  - `limit`: Maximum number of users to return (default: 10, max: 100)

**Response (200 OK)**:

```json
[
  {
    "_id": "507f1f77bcf86cd799439011",
    "name": "John Smith",
    "profile_picture": "https://example.com/profile.jpg",
    "workout_streak": 15,
    "total_workouts": 45,
    "badges": ["first_workout", "week_streak", "month_streak"]
  },
  {
    "_id": "507f1f77bcf86cd799439012",
    "name": "Jane Doe",
    "profile_picture": "https://example.com/profile2.jpg",
    "workout_streak": 12,
    "total_workouts": 36,
    "badges": ["first_workout", "week_streak"]
  }
  // More users...
]
```

**Error Responses**:

- `401 Unauthorized`: Not authenticated

## WebSocket Connection

### Connect to WebSocket

Connect to receive real-time analysis during workouts.

- **URL**: `/ws/{client_id}`
- **Protocol**: WebSocket
- **Parameters**:
  - `client_id`: Unique identifier for the client session

**WebSocket Messages**:

1. Connection Established:

```json
{
  "status": "connected",
  "message": "Connection established"
}
```

2. Heartbeat (sent every 15 seconds):

```json
{
  "type": "heartbeat"
}
```

3. Data Received Acknowledgment:

```json
{
  "status": "ok",
  "type": "data_received",
  "received_data_length": 1024,
  "message": "Data received successfully"
}
```

**Client Messages**:

Send base64-encoded video frames for analysis, or JSON control messages.

**Disconnection**:

The server will automatically disconnect inactive clients after a period of inactivity.

## Health Check

Check the API service status.

- **URL**: `/health`
- **Method**: `GET`

**Response (200 OK)**:

```json
{
  "status": "ok",
  "database": "connected",
  "mongo_status": "ok",
  "version": "0.1.0"
}
```
