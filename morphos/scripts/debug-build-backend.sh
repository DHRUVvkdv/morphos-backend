#!/bin/bash
set -e

# Store the script's location
SCRIPT_DIR="$(dirname "$0")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Only focus on backend service to save time
echo "Starting debug build process for backend service only..."

# Clean up old image
echo "Cleaning up old backend image..."
gcloud container images delete gcr.io/boxwood-veld-455217-p6/morphos-backend-service --quiet || true

# Build with debugging enabled
echo "Building backend service with debugging..."
cd "$PROJECT_ROOT/services/backend-service"

# Show contents of key files for debugging
echo "Current requirements.txt:"
cat requirements.txt

echo "Current Dockerfile:"
cat Dockerfile

# Build the image
gcloud builds submit --tag gcr.io/boxwood-veld-455217-p6/morphos-backend-service .

echo "Backend service build completed successfully!"
echo "Now run the debug-deploy.sh script to deploy it."