#!/bin/bash
set -e

# Store the script's location
SCRIPT_DIR="$(dirname "$0")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Build and push Inference service with GPU
echo "Building Inference service with GPU support..."
cd "$PROJECT_ROOT/services/inference-service"
gcloud builds submit --tag gcr.io/boxwood-veld-455217-p6/morphos-inference-service .

echo "Inference service build complete."