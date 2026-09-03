"""FastAPI web service: upload an audio file, get AudioSet tags back.

Run locally:
    uvicorn app.server:app --host 0.0.0.0 --port 8080

Configuration is via environment variables (see ``Settings``).
"""
import logging
import os
import tempfile
import time
from typing import Optional

import uuid

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .tagger import AudioTagger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("audiotagging")

HERE = os.path.dirname(os.path.abspath(__file__))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


class Settings:
    model_type = os.environ.get("MODEL_TYPE", "Cnn14")
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "/models/Cnn14_mAP=0.431.pth")
    sample_rate = _env_int("SAMPLE_RATE", 32000)
    window_size = _env_int("WINDOW_SIZE", 1024)
    hop_size = _env_int("HOP_SIZE", 320)
    mel_bins = _env_int("MEL_BINS", 64)
    fmin = _env_int("FMIN", 50)
    fmax = _env_int("FMAX", 14000)
    max_upload_mb = _env_int("MAX_UPLOAD_MB", 100)
    max_duration_seconds = _env_int("MAX_DURATION_SECONDS", 600)
    default_top_k = _env_int("DEFAULT_TOP_K", 10)
    num_threads = _env_int("TORCH_NUM_THREADS", 0) or None
    # If set, requests to /api/* must carry it in the X-API-Key header or ?key=
    api_key: Optional[str] = os.environ.get("API_KEY") or None
    # If set, browsers can upload large files straight to this GCS bucket and
    # then call /api/tag-object. Bypasses Cloud Run's 32 MiB request limit.
    upload_bucket: Optional[str] = os.environ.get("UPLOAD_BUCKET") or None
    upload_max_mb = _env_int("UPLOAD_MAX_MB", 1024)
    # Files above this size go through the bucket in the browser UI.
    direct_upload_max_mb = _env_int("DIRECT_UPLOAD_MAX_MB", 25)


settings = Settings()
app = FastAPI(title="Ajento Audio Tagging", version="1.0.0",
              description="AudioSet tagging with PANNs (Cnn14). Upload audio, get labels.")

tagger: Optional[AudioTagger] = None


@app.on_event("startup")
def _load_model() -> None:
    global tagger
    t0 = time.time()
    ckpt = settings.checkpoint_path if os.path.exists(settings.checkpoint_path) else None
    if ckpt is None:
        log.warning("Checkpoint %s not found; serving with RANDOM weights (dev only)",
                    settings.checkpoint_path)
    tagger = AudioTagger(
        model_type=settings.model_type,
        checkpoint_path=ckpt,
        sample_rate=settings.sample_rate,
        window_size=settings.window_size,
        hop_size=settings.hop_size,
        mel_bins=settings.mel_bins,
        fmin=settings.fmin,
        fmax=settings.fmax,
        num_threads=settings.num_threads,
    )
    log.info("Loaded %s on %s in %.1fs", settings.model_type, tagger.device, time.time() - t0)


def require_api_key(
    x_api_key: Optional[str] = Header(default=None),
    key: Optional[str] = Query(default=None),
) -> None:
    if settings.api_key and (x_api_key or key) != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"status": "ok", "model_loaded": tagger is not None,
            "weights": "pretrained" if tagger and tagger.checkpoint_path else "random"}


@app.get("/api/labels", dependencies=[Depends(require_api_key)])
def labels():
    return {"count": tagger.classes_num, "labels": tagger.labels}


def _tag_local_file(tmp_path: str, filename: str, size: int, top_k: int, threshold: float,
                    segment_seconds: float, include_embedding: bool, t0: float) -> dict:
    try:
        try:
            waveform = tagger.decode_audio(tmp_path, max_seconds=settings.max_duration_seconds)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        result = tagger.tag(waveform, top_k=top_k, threshold=threshold,
                            segment_seconds=segment_seconds, include_embedding=include_embedding)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    result["filename"] = filename
    result["truncated_to_seconds"] = (
        settings.max_duration_seconds
        if result["duration_seconds"] >= settings.max_duration_seconds - 0.05 else None
    )
    result["processing_seconds"] = round(time.time() - t0, 3)
    log.info("tagged %s (%.1fs audio, %d bytes) in %.2fs -> %s",
             filename, result["duration_seconds"], size, result["processing_seconds"],
             result["tags"][0]["label"] if result["tags"] else "-")
    return result


@app.get("/api/config")
def config():
    return {
        "direct_upload_max_mb": min(settings.max_upload_mb, settings.direct_upload_max_mb)
        if settings.upload_bucket else settings.max_upload_mb,
        "bucket_upload": settings.upload_bucket is not None,
        "upload_max_mb": settings.upload_max_mb if settings.upload_bucket else settings.max_upload_mb,
        "max_duration_seconds": settings.max_duration_seconds,
        "default_top_k": settings.default_top_k,
    }


@app.post("/api/tag", dependencies=[Depends(require_api_key)])
async def tag_audio(
    file: UploadFile = File(..., description="Audio file (wav, mp3, m4a, flac, ogg, webm...)"),
    top_k: int = Form(default=None, ge=1, le=527),
    threshold: float = Form(default=0.0, ge=0.0, le=1.0),
    segment_seconds: float = Form(default=0.0, ge=0.0, le=60.0,
                                  description="Also return tags per segment of this length (0 = off)"),
    include_embedding: bool = Form(default=False),
):
    if tagger is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    top_k = top_k or settings.default_top_k

    limit = settings.max_upload_mb * 1024 * 1024
    suffix = os.path.splitext(file.filename or "")[1][:10] or ".bin"
    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        size = 0
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                tmp.close()
                os.remove(tmp.name)
                raise HTTPException(status_code=413,
                                    detail=f"File exceeds {settings.max_upload_mb} MB limit")
            tmp.write(chunk)
        tmp_path = tmp.name
    if size == 0:
        os.remove(tmp_path)
        raise HTTPException(status_code=400, detail="Empty upload")
    return JSONResponse(_tag_local_file(tmp_path, file.filename, size, top_k, threshold,
                                        segment_seconds, include_embedding, t0))


# ---------------------------------------------------------------- large files
class UploadSessionRequest(BaseModel):
    filename: str
    content_type: str = "application/octet-stream"
    size: int


def _gcs_bucket():
    if not settings.upload_bucket:
        raise HTTPException(status_code=404, detail="Bucket uploads are not enabled")
    from google.cloud import storage  # imported lazily; optional dependency
    return storage.Client().bucket(settings.upload_bucket)


@app.post("/api/upload-session", dependencies=[Depends(require_api_key)])
def create_upload_session(req: UploadSessionRequest, request: Request):
    """Start a resumable GCS upload the browser can PUT the file to directly."""
    if req.size <= 0 or req.size > settings.upload_max_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.upload_max_mb} MB limit")
    suffix = os.path.splitext(req.filename or "")[1][:10]
    object_name = f"uploads/{uuid.uuid4().hex}{suffix}"
    blob = _gcs_bucket().blob(object_name)
    origin = request.headers.get("origin")
    try:
        url = blob.create_resumable_upload_session(
            content_type=req.content_type or "application/octet-stream",
            size=req.size, origin=origin)
    except Exception as e:  # surfaces missing IAM etc. as a clean 500
        log.exception("could not create upload session")
        raise HTTPException(status_code=500, detail=f"Could not start upload: {e}")
    return {"object": object_name, "upload_url": url}


@app.post("/api/tag-object", dependencies=[Depends(require_api_key)])
def tag_object(
    object_name: str = Form(..., alias="object"),
    filename: str = Form(default=""),
    top_k: int = Form(default=None, ge=1, le=527),
    threshold: float = Form(default=0.0, ge=0.0, le=1.0),
    segment_seconds: float = Form(default=0.0, ge=0.0, le=60.0),
    include_embedding: bool = Form(default=False),
):
    """Tag a file previously uploaded via /api/upload-session, then delete it."""
    if tagger is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if not object_name.startswith("uploads/") or "/" in object_name[len("uploads/"):]:
        raise HTTPException(status_code=400, detail="Invalid object name")
    top_k = top_k or settings.default_top_k
    t0 = time.time()
    blob = _gcs_bucket().blob(object_name)
    suffix = os.path.splitext(object_name)[1] or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
    try:
        blob.download_to_filename(tmp_path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=404, detail=f"Uploaded file not found: {e}")
    finally:
        try:
            blob.delete()
        except Exception:
            pass
    size = os.path.getsize(tmp_path)
    return JSONResponse(_tag_local_file(tmp_path, filename or os.path.basename(object_name), size,
                                        top_k, threshold, segment_seconds, include_embedding, t0))


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
