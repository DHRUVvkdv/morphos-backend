#!/bin/bash
set -e

# Build and push Backend service
echo "Building Morphos Backend service..."
cd "$(dirname "$0")/../services/backend-service"
gcloud builds submit --tag gcr.io/boxwood-veld-455217-p6/morphos-backend-service .

# Build and push Inference service 
# echo "Building Inference service..."
# cd "$(dirname "$0")/../services/inference-service"
# gcloud builds submit --tag gcr.io/boxwood-veld-455217-p6/morphos-inference-service .