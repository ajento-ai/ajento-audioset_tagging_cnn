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

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
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

    result["filename"] = file.filename
    result["truncated_to_seconds"] = (
        settings.max_duration_seconds
        if result["duration_seconds"] >= settings.max_duration_seconds - 0.05 else None
    )
    result["processing_seconds"] = round(time.time() - t0, 3)
    log.info("tagged %s (%.1fs audio, %d bytes) in %.2fs -> %s",
             file.filename, result["duration_seconds"], size, result["processing_seconds"],
             result["tags"][0]["label"] if result["tags"] else "-")
    return JSONResponse(result)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
