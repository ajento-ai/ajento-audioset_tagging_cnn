# Deploying the audio tagging service to GCP

This repo ships a small web service (`app/`) around the pretrained PANNs
**Cnn14** model. You upload an audio file and get back AudioSet labels with
probabilities, as JSON or through a browser UI. It is packaged with the
`Dockerfile` for **Cloud Run** and exposed on a custom domain such as
`audiotagging.ajento.app`.

```
Browser / curl  ──►  https://audiotagging.ajento.app
                        │  Cloud Run (2 vCPU, 4 GiB, scale 0→5)
                        │  FastAPI + ffmpeg + torch (CPU)
                        └─ Cnn14_mAP=0.431.pth baked into the image
```

## Prerequisites

- A GCP project with billing enabled and the `gcloud` CLI authenticated
  (`gcloud auth login && gcloud auth application-default login`).
- Permission to enable APIs and create Cloud Run / Artifact Registry / Cloud Build resources.
- Control over DNS for `ajento.app`.

## 1. Deploy

```bash
git clone <this repo> && cd ajento-audioset_tagging_cnn
PROJECT_ID=<your-gcp-project> ./deploy/deploy.sh
```

The script:

1. Enables Cloud Run, Cloud Build and Artifact Registry.
2. Builds the image with Cloud Build (the checkpoint, ~330 MB, is downloaded
   from Zenodo during the build so cold starts never hit the internet).
3. Deploys the Cloud Run service `audiotagging` (public, 2 vCPU, 4 GiB,
   concurrency 2, 300 s request timeout).
4. Creates the Cloud Run domain mapping for `audiotagging.ajento.app` and
   prints the DNS record you must add.

Useful overrides: `REGION`, `SERVICE`, `DOMAIN` (set `DOMAIN=` to skip the
mapping), `MIN_INSTANCES=1` (keeps one warm instance so the ~20 s model
load never hits a user), `API_KEY=<secret>` (requires `X-API-Key` on `/api/*`).

### Alternative: deploy from GitHub Actions

`.github/workflows/deploy.yml` runs the same script from CI. Create a service
account, grant it `roles/run.admin`, `roles/cloudbuild.builds.editor`,
`roles/artifactregistry.admin`, `roles/serviceusage.serviceUsageAdmin` and
`roles/iam.serviceAccountUser`, download a JSON key, and add it as the repo
secret `GCP_SA_KEY` plus the repo variable `GCP_PROJECT_ID`. Then trigger the
workflow from the Actions tab (or push to `master`).

```bash
PROJECT_ID=<your-gcp-project>
SA=deployer@${PROJECT_ID}.iam.gserviceaccount.com
gcloud iam service-accounts create deployer --project $PROJECT_ID
for r in run.admin cloudbuild.builds.editor artifactregistry.admin serviceusage.serviceUsageAdmin iam.serviceAccountUser storage.admin; do
  gcloud projects add-iam-policy-binding $PROJECT_ID --member serviceAccount:$SA --role roles/$r
done
gcloud iam service-accounts keys create key.json --iam-account $SA
```

## 2. Point the domain

Cloud Run domain mappings need the parent domain verified once for your Google
account. If the script reports that, run `gcloud domains verify ajento.app`,
complete the Search Console TXT verification, and re-run the script.

Then add the DNS record printed by the script. It is normally:

| Type  | Name           | Value                    |
|-------|----------------|--------------------------|
| CNAME | `audiotagging` | `ghs.googlehosted.com.`  |

Google provisions the TLS certificate automatically once DNS resolves
(usually 15 to 60 minutes). Check with:

```bash
gcloud beta run domain-mappings describe --domain audiotagging.ajento.app --region us-central1
```

If you would rather front the service with a global HTTPS load balancer
(Cloud Armor, IAP, multi-region), skip the mapping and follow
https://cloud.google.com/run/docs/multiple-regions instead.

## 3. Use it

Browser: open `https://audiotagging.ajento.app`, drop a file, click Analyze.

API:

```bash
curl -F "file=@clip.mp3" -F top_k=5 -F segment_seconds=10 \
  https://audiotagging.ajento.app/api/tag
```

```json
{
  "duration_seconds": 7.0,
  "model": "Cnn14",
  "tags": [
    {"index": 0, "label": "Speech", "probability": 0.893},
    {"index": 388, "label": "Telephone bell ringing", "probability": 0.754}
  ],
  "segments": [{"start": 0, "end": 7.0, "tags": [...]}],
  "filename": "clip.mp3",
  "processing_seconds": 1.2
}
```

Form fields for `POST /api/tag`:

| Field               | Default | Meaning                                              |
|---------------------|---------|------------------------------------------------------|
| `file`              |         | Any ffmpeg-decodable audio/video: wav, mp3, m4a, flac, ogg, webm |
| `top_k`             | 10      | Number of labels to return (1–527)                   |
| `threshold`         | 0.0     | Drop labels below this probability                   |
| `segment_seconds`   | 0       | Also tag each fixed-length window (0 disables)       |
| `include_embedding` | false   | Return the 2048-d clip embedding                     |

Large files: Cloud Run rejects request bodies over 32 MiB, so the browser UI
uploads bigger files straight to a Cloud Storage bucket and then calls
`POST /api/tag-object`. Programmatic flow: `POST /api/upload-session`
(JSON `filename`, `content_type`, `size`) returns an `upload_url` to `PUT` the
file to and an `object` name to pass to `/api/tag-object`. Objects are deleted
after tagging and the bucket auto-expires anything left after one day. The
Cloud Run runtime service account needs `roles/storage.objectAdmin` on the
bucket (the deploy script prints a reminder).

Recordings longer than 60 s are processed in 10 s windows and the overall tags
are the mean over windows (`"aggregation": "mean_over_segments"`), which keeps
memory flat for long files.

### Timestamped timeline and events

A second, frame-level model (`Cnn14_DecisionLevelMax`) is baked into the image
alongside the main one and gives a probability per class roughly every 10 ms,
instead of one score for the whole clip. Pass `timeline_seconds` (0 to 10,
0 = off) to `/api/tag` or `/api/tag-object` to get, on top of the normal
result:

- `timeline`: fixed-size windows of that many seconds, each with its own top
  tags — good for scrubbing through the file.
- `events`: discrete start/end/peak-time ranges per label wherever its
  probability stayed above the threshold, merged across short gaps. This is
  what answers "what happened, and when" — e.g. a `Slam` at 45.3s or `Music`
  running from 12s to 90s.

It still only names AudioSet's 527 generic classes with a timestamp (`Slam`,
`Thud`, `Bang`, `Music`, `Speech`, ...), not a description of the scene — it
cannot tell you "she hits wall", only that an impact-like sound happened at
that time. Set `SED_MODEL_URL=""` at build time to skip downloading this
second checkpoint if you don't need the feature; `GET /api/config` reports
`timeline_available` so the UI can hide the option automatically.

Other endpoints: `GET /api/config`, `GET /api/labels`, `GET /healthz`, `GET /docs` (OpenAPI UI).
Direct uploads are capped at 32 MB, bucket uploads at 1 GB, and analysis at
the first 30 minutes of audio (env `MAX_UPLOAD_MB`, `UPLOAD_MAX_MB`,
`MAX_DURATION_SECONDS`).

## Running locally

```bash
docker build -t audiotagging .
docker run --rm -p 8080:8080 audiotagging
open http://localhost:8080
```

Without Docker (needs ffmpeg on PATH for non-wav input):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r app/requirements.txt
wget -O Cnn14_mAP=0.431.pth "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
CHECKPOINT_PATH=Cnn14_mAP=0.431.pth uvicorn app.server:app --port 8080
```

## Swapping the model

Any audio-tagging model in `pytorch/models.py` works. For the 16 kHz model:

```bash
gcloud builds submit --config deploy/cloudbuild.yaml .  # after editing the ARGs below
```

Set Docker build args `MODEL_URL` / `MODEL_FILE` and runtime env
`MODEL_TYPE=Cnn14_16k SAMPLE_RATE=16000 WINDOW_SIZE=512 HOP_SIZE=160 FMAX=8000`.

## Cost and performance notes

- CPU inference of a 10 s clip on 2 vCPU takes roughly 1 to 2 s; a 10 minute
  file takes about a minute. Use `--cpu=4` for faster turnaround.
- Scale-to-zero costs nothing idle but the first request after idle waits
  ~20 s for the model to load. `MIN_INSTANCES=1` removes that for roughly
  $50 per month.
- Cloud Run rejects request bodies over 32 MiB with an HTML 413 page before
  the app sees them. A 32 MB MP3 is roughly 30 minutes of audio, well above
  the 600 s the service analyses, so compress long recordings before upload.
