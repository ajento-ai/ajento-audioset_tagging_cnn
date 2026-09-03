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

Other endpoints: `GET /api/labels`, `GET /healthz`, `GET /docs` (OpenAPI UI).
Uploads are capped at 100 MB and the first 600 s of audio (env
`MAX_UPLOAD_MB`, `MAX_DURATION_SECONDS`).

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
- Cloud Run has a 32 MiB request limit only for HTTP/1 without chunking; file
  uploads up to the `MAX_UPLOAD_MB` limit work over HTTP/2, which Cloud Run
  uses by default.
