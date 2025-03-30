#!/bin/bash
set -e

# Store the script's location
SCRIPT_DIR="$(dirname "$0")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Clean up old image first
echo "Cleaning up old backend image..."
gcloud container images delete gcr.io/boxwood-veld-455217-p6/morphos-backend-service --quiet || true

# Build with specific tag to avoid caching issues
echo "Building Morphos Backend service..."
cd "$PROJECT_ROOT/services/backend-service"
gcloud builds submit --tag gcr.io/boxwood-veld-455217-p6/morphos-backend-service:$(date +%s) .

# Deploy the service
echo "Deploying Morphos Backend Service..."
gcloud run deploy morphos-backend-service \
  --image gcr.io/boxwood-veld-455217-p6/morphos-backend-service:$(date +%s) \
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
  --set-env-vars="INFERENCE_SERVICE_URL=https://morphos-inference-service-s4uldl3cvq-uc.a.run.app,DEBUG=true" \
  --cpu-throttling

echo "Deployment completed. Checking logs..."
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=morphos-backend-service" --limit=20