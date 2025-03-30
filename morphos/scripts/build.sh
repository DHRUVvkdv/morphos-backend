#!/bin/bash
set -e

# Store the script's location
SCRIPT_DIR="$(dirname "$0")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Build and push Backend service
echo "Building Morphos Backend service..."
cd "$PROJECT_ROOT/services/backend-service"
gcloud builds submit --tag gcr.io/boxwood-veld-455217-p6/morphos-backend-service .

# Build and push Inference service 
echo "Building Inference service..."
cd "$PROJECT_ROOT/services/inference-service"
gcloud builds submit --tag gcr.io/boxwood-veld-455217-p6/morphos-inference-service .