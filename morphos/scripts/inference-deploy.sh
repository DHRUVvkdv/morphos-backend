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

# Deploy Inference Service with WebSocket support and session affinity
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
  --set-env-vars="PYTHONUNBUFFERED=1,NVIDIA_VISIBLE_DEVICES=all" \
  --gpu 1 \
  --no-cpu-boost \
  --service-account="1020595365432-compute@developer.gserviceaccount.com" \
  --session-affinity \
  --ingress=all

# Test the service after deployment
echo "Testing service health..."
INFERENCE_URL=$(gcloud run services describe morphos-inference-service --platform managed --region us-central1 --format 'value(status.url)')
echo "Service URL: $INFERENCE_URL"

# Try health check a few times with backoff
for i in {1..5}; do
  echo "Health check attempt $i..."
  if curl -s "$INFERENCE_URL/health" | grep -q "ok"; then
    echo "Service is healthy!"
    break
  else
    echo "Service not ready yet, waiting..."
    sleep $((5 * i))
  fi
done

echo "Deployment complete. Service URL: $INFERENCE_URL"