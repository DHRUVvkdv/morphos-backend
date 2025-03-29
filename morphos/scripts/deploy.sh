#!/bin/bash
set -e

# Deploy WebSocket service
echo "Deploying WebSocket service..."
gcloud run deploy morphos-backend-service \
  --image gcr.io/boxwood-veld-455217-p6/morphos-backend-service \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --cpu 2 \
  --memory 4Gi \
  --timeout 3600 \
  --concurrency 80 \
  --min-instances 0 \
  --max-instances 10 \
  --region us-central1 \
  --set-env-vars="ENVIRONMENT=production" \
  --use-http2

# Deploy Inference service (when ready)
# echo "Deploying Inference service..."
# gcloud run deploy morphos-inference-service \
#   --image gcr.io/YOUR_PROJECT_ID/morphos-inference-service \
#   --platform managed \
#   --allow-unauthenticated \
#   --port 8000 \
#   --cpu 4 \
#   --memory 16Gi \
#   --timeout 3600 \
#   --concurrency 40 \
#   --min-instances 0 \
#   --max-instances 2 \
#   --region us-central1 \
#   --gpu 1 \
#   --gpu-type=nvidia-t4