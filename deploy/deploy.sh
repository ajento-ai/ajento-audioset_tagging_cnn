#!/usr/bin/env bash
# One-shot deploy of the audio tagging service to Google Cloud Run.
#
#   PROJECT_ID=my-gcp-project ./deploy/deploy.sh
#
# Optional env vars:
#   REGION      (default us-central1)  Cloud Run region (must support domain mappings)
#   SERVICE     (default audiotagging) Cloud Run service name
#   DOMAIN      (default audiotagging.ajento.app) custom domain; set empty to skip
#   API_KEY     (default unset)        if set, /api/* requires X-API-Key header
#   MIN_INSTANCES (default 0)          set to 1 to avoid ~20 s cold starts
#   MEMORY / CPU (default 4Gi / 2)
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your GCP project id}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-audiotagging}"
REPO="${REPO:-audiotagging}"
DOMAIN="${DOMAIN-audiotagging.ajento.app}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-5}"
MEMORY="${MEMORY:-8Gi}"  # tagging + timeline/events + speech-to-text models
CPU="${CPU:-4}"
# Bucket for browser uploads larger than Cloud Run's 32 MiB request limit.
# Set UPLOAD_BUCKET= (empty) to disable.
UPLOAD_BUCKET="${UPLOAD_BUCKET-${PROJECT_ID}-${SERVICE}-uploads}"
MAX_DURATION_SECONDS="${MAX_DURATION_SECONDS:-1800}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}"

cd "$(dirname "$0")/.."

echo "==> Project ${PROJECT_ID}, region ${REGION}, service ${SERVICE}"
gcloud config set project "${PROJECT_ID}" >/dev/null

echo "==> Enabling APIs"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

echo "==> Ensuring Artifact Registry repo ${REPO}"
if ! gcloud artifacts repositories describe "${REPO}" --location="${REGION}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPO}" --repository-format=docker --location="${REGION}" \
    --description="Audio tagging service images"
fi

# Newer projects give the default compute service account no roles, which makes
# Cloud Build fail with "does not have storage.objects.get access". Fix once with:
#   gcloud projects add-iam-policy-binding $PROJECT_ID \
#     --member serviceAccount:$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')-compute@developer.gserviceaccount.com \
#     --role roles/cloudbuild.builds.builder
# or set BUILD_SERVICE_ACCOUNT to a user-managed SA that has that role.
BUILD_SA_FLAG=()
if [[ -n "${BUILD_SERVICE_ACCOUNT:-}" ]]; then
  BUILD_SA_FLAG=(--service-account="projects/${PROJECT_ID}/serviceAccounts/${BUILD_SERVICE_ACCOUNT}")
fi

echo "==> Building image with Cloud Build (downloads the ~330 MB checkpoint; takes several minutes)"
gcloud builds submit --config deploy/cloudbuild.yaml "${BUILD_SA_FLAG[@]}" \
  --substitutions="_REGION=${REGION},_REPO=${REPO},_IMAGE=${SERVICE},SHORT_SHA=$(git rev-parse --short HEAD 2>/dev/null || date +%s)" .

ENV_VARS="MODEL_TYPE=Cnn14,MAX_UPLOAD_MB=32,MAX_DURATION_SECONDS=${MAX_DURATION_SECONDS},TORCH_NUM_THREADS=${CPU}"
# Interpretation column via Gemini on Vertex AI, using the runtime service
# account (needs roles/aiplatform.user). The 3.x flash models resolve only in
# the "global" location.
ENV_VARS="${ENV_VARS},GEMINI_PROJECT=${PROJECT_ID},GEMINI_MODEL=${GEMINI_MODEL:-gemini-3.8-flash},GEMINI_LOCATION=${GEMINI_LOCATION:-global}"
if [[ -n "${API_KEY:-}" ]]; then ENV_VARS="${ENV_VARS},API_KEY=${API_KEY}"; fi

if [[ -n "${UPLOAD_BUCKET}" ]]; then
  echo "==> Ensuring upload bucket gs://${UPLOAD_BUCKET}"
  if ! gcloud storage buckets describe "gs://${UPLOAD_BUCKET}" >/dev/null 2>&1; then
    gcloud storage buckets create "gs://${UPLOAD_BUCKET}" --location="${REGION}" --uniform-bucket-level-access
  fi
  TMP_CFG="$(mktemp -d)"
  ORIGINS="\"https://${SERVICE}-$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)').${REGION}.run.app\", \"http://localhost:8080\""
  if [[ -n "${DOMAIN}" ]]; then ORIGINS="\"https://${DOMAIN}\", ${ORIGINS}"; fi
  cat > "${TMP_CFG}/cors.json" <<EOF2
[{"origin": [${ORIGINS}], "method": ["PUT", "POST", "OPTIONS"],
  "responseHeader": ["Content-Type", "Content-Range", "x-goog-resumable", "Location", "Range"], "maxAgeSeconds": 3600}]
EOF2
  echo '{"rule": [{"action": {"type": "Delete"}, "condition": {"age": 1}}]}' > "${TMP_CFG}/lifecycle.json"
  gcloud storage buckets update "gs://${UPLOAD_BUCKET}" --cors-file="${TMP_CFG}/cors.json" --lifecycle-file="${TMP_CFG}/lifecycle.json"
  ENV_VARS="${ENV_VARS},UPLOAD_BUCKET=${UPLOAD_BUCKET}"
  echo "   The Cloud Run runtime service account needs roles/storage.objectAdmin on this bucket."
fi

# --no-invoker-iam-check makes the service public without an allUsers IAM
# binding, which organizations with domain-restricted sharing reject.
echo "==> Deploying to Cloud Run"
gcloud run deploy "${SERVICE}" \
  --image="${IMAGE}:latest" \
  --region="${REGION}" \
  --platform=managed \
  --no-invoker-iam-check \
  --port=8080 \
  --memory="${MEMORY}" --cpu="${CPU}" \
  --concurrency=2 \
  --timeout=900 \
  --min-instances="${MIN_INSTANCES}" --max-instances="${MAX_INSTANCES}" \
  --cpu-boost \
  --set-env-vars="${ENV_VARS}"

URL="$(gcloud run services describe "${SERVICE}" --region="${REGION}" --format='value(status.url)')"
echo "==> Service URL: ${URL}"

if [[ -n "${DOMAIN}" ]]; then
  echo "==> Mapping custom domain ${DOMAIN}"
  if ! gcloud beta run domain-mappings describe --domain="${DOMAIN}" --region="${REGION}" >/dev/null 2>&1; then
    gcloud beta run domain-mappings create --service="${SERVICE}" --domain="${DOMAIN}" --region="${REGION}" || {
      echo "!! Domain mapping failed. The parent domain must be verified for this account:"
      echo "   gcloud domains verify ajento.app   (opens Search Console)"
      echo "   then re-run this script."
    }
  fi
  echo "==> Add this DNS record at your DNS provider for ajento.app (then wait for the managed cert, ~15 min):"
  gcloud beta run domain-mappings describe --domain="${DOMAIN}" --region="${REGION}" \
    --format='table(status.resourceRecords[].type, status.resourceRecords[].name, status.resourceRecords[].rrdata)' || true
  echo "   (typically: CNAME  ${DOMAIN%%.*}  ->  ghs.googlehosted.com.)"
fi

echo
echo "Test:  curl -F file=@resources/R9_ZSCveAHg_7s.wav ${URL}/api/tag"
