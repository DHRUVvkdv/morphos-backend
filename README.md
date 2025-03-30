# morphos-backend

Morphos (Μορφή)

works in python 3.9 only

# Morphos - AI Workout Analysis App

Real-time workout analysis application using computer vision to track exercise form, detect emotions, and provide personalized feedback.

## Architecture

- **Frontend**: Next.js/React application on Vercel
- **Backend**: FastAPI application on Google Cloud Run
- **ML Models**: NVIDIA Dynamo (formerly Triton) on Google Cloud Run with GPU
- **Database**: MongoDB Atlas

## Project Structure

morphos/
├── services/ # Microservices
│ ├── websocket-service/ # WebSocket service
│ ├── inference-service/ # Dynamo inference service
│ └── api-service/ # Main REST API service
├── infra/ # Infrastructure as Code
├── scripts/ # Utility scripts
└── docker-compose.yaml # Local development setup

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Google Cloud SDK
- Python 3.9+

### Local Development

1. Clone the repository
2. Run the services locally:

```bash
docker-compose up
```

Access the WebSocket service at http://localhost:8080

Deployment
To deploy to Google Cloud:

# Build and push Docker images

./scripts/build.sh

# Deploy to Google Cloud Run

./scripts/deploy.sh

docker compose up --build

docker build -t morphos-inference-service .
docker run -p 8000:8000 -v "$(pwd)/models:/models" morphos-inference-service
