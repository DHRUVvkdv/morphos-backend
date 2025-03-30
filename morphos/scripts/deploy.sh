#!/bin/bash
set -e

# Create the NVIDIA driver capabilities secret (or update if it exists)
echo "Creating NVIDIA driver capabilities secret..."
if gcloud secrets describe nvidia-driver-capabilities &>/dev/null; then
  echo "Secret already exists, updating version..."
  echo -n "compute,utility" | gcloud secrets versions add nvidia-driver-capabilities --data-file=-
else
  echo "Creating new secret..."
  echo -n "compute,utility" | gcloud secrets create nvidia-driver-capabilities --replication-policy="automatic" --data-file=-
fi

# Grant the Cloud Run service account access to the secret
echo "Granting service account access to the secret..."
SERVICE_ACCOUNT="service-1020595365432@serverless-robot-prod.iam.gserviceaccount.com"
gcloud secrets add-iam-policy-binding nvidia-driver-capabilities \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor"

# Deploy Inference Service
echo "Deploying Morphos Inference Service..."
gcloud run deploy morphos-inference-service \
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
  --set-secrets="NVIDIA_DRIVER_CAPABILITIES=nvidia-driver-capabilities:latest" \
  --gpu 1

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