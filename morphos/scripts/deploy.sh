#!/bin/bash
set -e
# Deploy Inference Service with GPU
echo "Deploying Morphos Inference Service..."
gcloud beta run deploy morphos-inference-service \
  --image gcr.io/boxwood-veld-455217-p6/morphos-inference-service \
  --platform managed \
  --allow-unauthenticated \
  --port 8000 \
  --timeout 3600 \
  --concurrency 40 \
  --cpu 4 \
  --memory 16Gi \
  --min-instances 0 \
  --max-instances 2 \
  --region us-central1 \
  --execution-environment gen2 \
  --gpu 1 \
  --gpu-type nvidia-l4

# Get the Inference Service URL
INFERENCE_URL=$(gcloud run services describe morphos-inference-service --platform managed --region us-central1 --format 'value(status.url)')

# Deploy Backend Service
echo "Deploying Morphos Backend Service with Inference URL: $INFERENCE_URL"
gcloud run deploy morphos-backend-service \
  --image gcr.io/boxwood-veld-455217-p6/morphos-backend-service \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --timeout 3600 \
  --concurrency 80 \
  --cpu 1 \
  --memory 512Mi \
  --min-instances 0 \
  --max-instances 10 \
  --region us-central1 \
  --execution-environment gen2 \
  --session-affinity \
  --set-env-vars="INFERENCE_SERVICE_URL=$INFERENCE_URL"