#!/bin/bash
set -e

# Store the script's location
SCRIPT_DIR="$(dirname "$0")"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Generate a timestamp for consistent image tagging
TIMESTAMP=$(date +%s)
IMAGE_TAG="gcr.io/boxwood-veld-455217-p6/morphos-backend-service:$TIMESTAMP"

echo "Building and deploying backend with tag: $IMAGE_TAG"

# Build with specific tag
echo "Building Morphos Backend service..."
cd "$PROJECT_ROOT/services/backend-service"
gcloud builds submit --tag $IMAGE_TAG .

# Verify the image exists before deploying
echo "Verifying image exists in registry..."
max_retries=10
count=0
while [ $count -lt $max_retries ]; do
  if gcloud container images describe $IMAGE_TAG >/dev/null 2>&1; then
    echo "Image found in registry. Proceeding with deployment."
    break
  else
    echo "Waiting for image to be available in registry... (Attempt $((count+1))/$max_retries)"
    sleep 10
    count=$((count+1))
  fi
done

if [ $count -eq $max_retries ]; then
  echo "Error: Image not found in registry after $max_retries attempts. Aborting deployment."
  exit 1
fi

# Deploy the service with the verified image
echo "Deploying Morphos Backend Service..."
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
  --set-secrets="AUTH0_DOMAIN=AUTH0_DOMAIN:latest,AUTH0_CLIENT_ID=AUTH0_CLIENT_ID:latest,AUTH0_CLIENT_SECRET=AUTH0_CLIENT_SECRET:latest,AUTH0_AUDIENCE=AUTH0_AUDIENCE:latest,MONGODB_URI=MONGODB_URI:latest,API_KEY=API_KEY:latest" \
  --set-env-vars="INFERENCE_SERVICE_URL=https://morphos-inference-service-s4uldl3cvq-uc.a.run.app,DEBUG=true"

echo "Deployment completed. Checking logs..."