"""FastAPI web service: upload an audio file, get AudioSet tags back.

Run locally:
    uvicorn app.server:app --host 0.0.0.0 --port 8080

Configuration is via environment variables (see ``Settings``).
"""
import json
import logging
import os
import tempfile
import time
from typing import Optional

import uuid

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .tagger import AudioTagger
from .transcriber import SpeechTranscriber

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("audiotagging")

HERE = os.path.dirname(os.path.abspath(__file__))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


class Settings:
    model_type = os.environ.get("MODEL_TYPE", "Cnn14")
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "/models/Cnn14_mAP=0.431.pth")
    # Optional frame-level model for the timestamped timeline/events feature.
    sed_model_type = os.environ.get("SED_MODEL_TYPE", "Cnn14_DecisionLevelMax")
    sed_checkpoint_path = os.environ.get(
        "SED_CHECKPOINT_PATH", "/models/Cnn14_DecisionLevelMax_mAP=0.385.pth")
    # Speech-to-text for the transcript table. Set WHISPER_MODEL_SIZE="" to disable.
    whisper_model_size = os.environ.get("WHISPER_MODEL_SIZE", "base")
    whisper_model_dir = os.environ.get("WHISPER_MODEL_DIR", "/models/whisper")
    whisper_beam_size = _env_int("WHISPER_BEAM_SIZE", 1)
    # Speaker labels for the breakdown table. Set DIARIZATION="0" to disable.
    diarization_enabled = os.environ.get("DIARIZATION", "1") not in ("0", "false", "")
    # Interpretation column (inference, via Gemini). Uses Vertex AI with the
    # runtime service account by default; GEMINI_API_KEY switches to the API.
    # Set INTERPRETATION=0 to turn the column off entirely.
    interpretation_enabled = os.environ.get("INTERPRETATION", "1") not in ("0", "false", "")
    gemini_api_key: Optional[str] = os.environ.get("GEMINI_API_KEY") or None
    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
    gemini_project: Optional[str] = (os.environ.get("GEMINI_PROJECT")
                                     or os.environ.get("GOOGLE_CLOUD_PROJECT") or None)
    gemini_location = os.environ.get("GEMINI_LOCATION", "global")
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
transcriber: Optional[SpeechTranscriber] = None
diarizer = None
interpreter = None


@app.on_event("startup")
def _load_model() -> None:
    global tagger, transcriber, diarizer, interpreter
    t0 = time.time()
    ckpt = settings.checkpoint_path if os.path.exists(settings.checkpoint_path) else None
    if ckpt is None:
        log.warning("Checkpoint %s not found; serving with RANDOM weights (dev only)",
                    settings.checkpoint_path)
    sed_ckpt = settings.sed_checkpoint_path if os.path.exists(settings.sed_checkpoint_path) else None
    if sed_ckpt is None:
        log.warning("SED checkpoint %s not found; timestamped timeline/events disabled",
                    settings.sed_checkpoint_path)
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
        sed_model_type=settings.sed_model_type,
        sed_checkpoint_path=sed_ckpt,
    )
    if settings.whisper_model_size:
        try:
            transcriber = SpeechTranscriber(
                model_size=settings.whisper_model_size,
                download_root=settings.whisper_model_dir or None,
                beam_size=settings.whisper_beam_size,
            )
        except Exception:
            log.exception("Could not load Whisper model; transcript table disabled")
            transcriber = None

    if settings.diarization_enabled:
        try:
            from .diarizer import SpeakerDiarizer

            diarizer = SpeakerDiarizer(
                distance_threshold=float(os.environ.get("DIARIZATION_THRESHOLD", "0.35")))
        except Exception:
            log.exception("Could not load diarizer; speaker labels disabled")
            diarizer = None

    if settings.interpretation_enabled:
        try:
            from .interpreter import Interpreter

            interpreter = Interpreter(api_key=settings.gemini_api_key,
                                      model=settings.gemini_model,
                                      project=settings.gemini_project,
                                      location=settings.gemini_location)
        except Exception:
            log.exception("Could not init interpreter; interpretation column disabled")
            interpreter = None

    log.info("Loaded %s (+SED=%s, +Whisper=%s, +Diarizer=%s, +Interpreter=%s) on %s in %.1fs",
             settings.model_type, bool(tagger.sed_model), bool(transcriber), bool(diarizer),
             bool(interpreter), tagger.device, time.time() - t0)


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
                    segment_seconds: float, include_embedding: bool, timeline_seconds: float,
                    transcript_table: bool, breakdown: bool, t0: float,
                    progress=None, script_elements=None, num_speakers=None) -> dict:
    def step(stage: str, fraction: float) -> None:
        if progress:
            progress(stage, fraction)

    # One frame-level pass shared by the timeline and the breakdown table.
    cache: dict = {}
    try:
        step("decoding", 0.05)
        try:
            waveform = tagger.decode_audio(tmp_path, max_seconds=settings.max_duration_seconds)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        step("tagging", 0.15)
        result = tagger.tag(waveform, top_k=top_k, threshold=threshold,
                            segment_seconds=segment_seconds, include_embedding=include_embedding)
        if timeline_seconds:
            step("detecting events", 0.4)
            events_out = tagger.detect_events(waveform, bin_seconds=timeline_seconds,
                                              top_k=min(top_k, 5), threshold=max(threshold, 0.15),
                                              cache=cache)
            if events_out is None:
                result["timeline_unavailable"] = (
                    "Timestamped timeline is not available: the server has no "
                    "sound event detection model loaded."
                )
            else:
                result.update(events_out)
        if transcript_table:
            step("transcribing", 0.5)
            rows = tagger.build_transcript_table(waveform, transcriber,
                                                 threshold=max(threshold, 0.15), cache=cache)
            if rows is None:
                result["transcript_unavailable"] = (
                    "Transcript table is not available: the server needs both the "
                    "sound event detection model and the speech-to-text model."
                )
            else:
                result["transcript_table"] = rows
        if breakdown:
            step("building breakdown", 0.55)
            sub = lambda stage, frac: step(stage, 0.55 + 0.3 * frac)
            if script_elements:
                out = tagger.build_script_breakdown(waveform, script_elements, transcriber,
                                                    cache=cache, progress=sub)
                if out is None:
                    result["breakdown_unavailable"] = (
                        "Script alignment needs the sound event detection and speech-to-text models.")
                    rows, meta = None, None
                else:
                    rows = out.pop("rows")
                    meta = out
            else:
                rows = tagger.build_breakdown_table(
                    waveform, transcriber=transcriber, diarizer=diarizer, cache=cache,
                    progress=sub, num_speakers=num_speakers)
                meta = None
                if rows is None:
                    result["breakdown_unavailable"] = (
                        "Breakdown table needs the sound event detection model, which is not loaded.")
            if rows is not None:
                for row in rows:
                    for f in ("interpretation", "action", "performance", "camera"):
                        row.setdefault(f, "")
                if interpreter is not None and rows:
                    step("writing direction", 0.9)
                    scene = meta.get("scene") if meta else None
                    for row, suggestion in zip(rows, interpreter.annotate(
                            rows, scene=scene, script_mode=bool(script_elements))):
                        for f, value in suggestion.items():
                            if value and not (f == "action" and row.get("kind") == "direction"):
                                row[f] = value
                result["breakdown_table"] = rows
                if meta:
                    result["script"] = meta
                    result["breakdown_columns"] = {
                        "mode": "script",
                        "authored": ["dialogue", "entity", "notes", "action(direction rows)"],
                        "detected": ["start", "end", "audio", "heard", "alignment", "measured"],
                        "suggested": (["interpretation", "camera", "performance", "action(line rows)"]
                                      if interpreter is not None else []),
                    }
                else:
                    result["breakdown_columns"] = {
                        "mode": "detected",
                        "detected": ["start", "end", "dialogue", "audio", "entity",
                                     "confidence", "measured"],
                        "suggested": (["interpretation", "action", "performance", "camera"]
                                      if interpreter is not None else []),
                    }
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
        "timeline_available": tagger is not None and tagger.sed_model is not None,
        "transcript_available": (tagger is not None and tagger.sed_model is not None
                                 and transcriber is not None),
        "breakdown_available": tagger is not None and tagger.sed_model is not None,
        "diarization_available": diarizer is not None,
        "interpretation_available": interpreter is not None,
    }


async def _read_script(upload: Optional[UploadFile]):
    """Optional script (.pdf or text) -> parsed elements, or None."""
    if upload is None or not upload.filename:
        return None
    from .script_parser import parse_script

    data = await upload.read()
    if not data:
        return None
    if upload.filename.lower().endswith(".pdf") or data[:5] == b"%PDF-":
        import io
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read script PDF: {e}")
    else:
        text = data.decode("utf-8", errors="replace")
    elements = parse_script(text)
    if not any(e.kind == "line" for e in elements):
        raise HTTPException(status_code=400,
                            detail="Script has no dialogue lines I could recognise "
                                   "(expected 'NAME:' cues with dialogue beneath).")
    return elements


async def _spool_upload(file: UploadFile):
    """Stream an upload to a temp file, enforcing the size cap. -> (path, size, name)"""
    limit = settings.max_upload_mb * 1024 * 1024
    suffix = os.path.splitext(file.filename or "")[1][:10] or ".bin"
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
    return tmp_path, size, file.filename


def _fetch_object(object_name: str):
    """Download a bucket upload to a temp file and delete the object. -> (path, size)"""
    if not object_name.startswith("uploads/") or "/" in object_name[len("uploads/"):]:
        raise HTTPException(status_code=400, detail="Invalid object name")
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
    return tmp_path, os.path.getsize(tmp_path)


@app.post("/api/tag", dependencies=[Depends(require_api_key)])
async def tag_audio(
    file: UploadFile = File(..., description="Audio file (wav, mp3, m4a, flac, ogg, webm...)"),
    top_k: int = Form(default=None, ge=1, le=527),
    threshold: float = Form(default=0.0, ge=0.0, le=1.0),
    segment_seconds: float = Form(default=0.0, ge=0.0, le=60.0,
                                  description="Also return tags per segment of this length (0 = off)"),
    include_embedding: bool = Form(default=False),
    timeline_seconds: float = Form(default=0.0, ge=0.0, le=10.0,
                                   description="Also return a timestamped timeline/events at this bin size in seconds (0 = off)"),
    transcript_table: bool = Form(default=False,
                                  description="Also return one row per speech turn with its transcript (slower)"),
    breakdown: bool = Form(default=False,
                           description="Also return the production breakdown table (speech turns + events)"),
    script: Optional[UploadFile] = File(default=None,
                                        description="Optional script (.pdf/.txt); aligns the recording to it"),
    speakers: Optional[int] = Form(default=None, ge=1, le=12,
                                   description="Known number of speakers (without a script)"),
):
    if tagger is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    top_k = top_k or settings.default_top_k

    t0 = time.time()
    elements = await _read_script(script)
    tmp_path, size, _ = await _spool_upload(file)
    return JSONResponse(_tag_local_file(tmp_path, file.filename, size, top_k, threshold,
                                        segment_seconds, include_embedding, timeline_seconds,
                                        transcript_table, breakdown, t0,
                                        script_elements=elements, num_speakers=speakers))


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
async def tag_object(
    object_name: str = Form(..., alias="object"),
    filename: str = Form(default=""),
    top_k: int = Form(default=None, ge=1, le=527),
    threshold: float = Form(default=0.0, ge=0.0, le=1.0),
    segment_seconds: float = Form(default=0.0, ge=0.0, le=60.0),
    include_embedding: bool = Form(default=False),
    timeline_seconds: float = Form(default=0.0, ge=0.0, le=10.0),
    transcript_table: bool = Form(default=False),
    breakdown: bool = Form(default=False),
    script: Optional[UploadFile] = File(default=None),
    speakers: Optional[int] = Form(default=None, ge=1, le=12),
):
    """Tag a file previously uploaded via /api/upload-session, then delete it."""
    if tagger is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if not object_name.startswith("uploads/") or "/" in object_name[len("uploads/"):]:
        raise HTTPException(status_code=400, detail="Invalid object name")
    top_k = top_k or settings.default_top_k
    t0 = time.time()
    elements = await _read_script(script)
    tmp_path, size = _fetch_object(object_name)
    return JSONResponse(_tag_local_file(tmp_path, filename or os.path.basename(object_name), size,
                                        top_k, threshold, segment_seconds, include_embedding,
                                        timeline_seconds, transcript_table, breakdown, t0,
                                        script_elements=elements, num_speakers=speakers))


# ------------------------------------------------------------------ streaming
def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _analysis_stream(run):
    """Run a blocking analysis in a worker thread, streaming progress as SSE.

    The work is CPU-bound and takes a minute or two, so the browser needs to
    see stages rather than a spinner that might be a hung request.
    """
    import queue
    import threading

    updates: "queue.Queue" = queue.Queue()

    def progress(stage: str, fraction: float) -> None:
        updates.put(("progress", {"stage": stage, "fraction": round(float(fraction), 3)}))

    def worker() -> None:
        try:
            updates.put(("result", run(progress)))
        except HTTPException as e:
            updates.put(("error", {"detail": e.detail, "status": e.status_code}))
        except Exception as e:
            log.exception("Analysis failed")
            updates.put(("error", {"detail": str(e), "status": 500}))
        finally:
            updates.put((None, None))

    threading.Thread(target=worker, daemon=True).start()

    def generate():
        yield _sse("progress", {"stage": "queued", "fraction": 0.0})
        while True:
            try:
                kind, payload = updates.get(timeout=15)
            except queue.Empty:
                yield ": keep-alive\n\n"   # stop proxies closing an idle stream
                continue
            if kind is None:
                break
            yield _sse(kind, payload)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/tag-stream", dependencies=[Depends(require_api_key)])
async def tag_audio_stream(
    file: UploadFile = File(...),
    top_k: int = Form(default=None, ge=1, le=527),
    threshold: float = Form(default=0.0, ge=0.0, le=1.0),
    segment_seconds: float = Form(default=0.0, ge=0.0, le=60.0),
    include_embedding: bool = Form(default=False),
    timeline_seconds: float = Form(default=0.0, ge=0.0, le=10.0),
    transcript_table: bool = Form(default=False),
    breakdown: bool = Form(default=False),
    script: Optional[UploadFile] = File(default=None),
    speakers: Optional[int] = Form(default=None, ge=1, le=12),
):
    """Same as /api/tag, but streams progress events while it works."""
    if tagger is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    elements = await _read_script(script)
    tmp_path, size, filename = await _spool_upload(file)
    t0 = time.time()
    k = top_k or settings.default_top_k
    return _analysis_stream(lambda progress: _tag_local_file(
        tmp_path, filename, size, k, threshold, segment_seconds, include_embedding,
        timeline_seconds, transcript_table, breakdown, t0, progress,
        script_elements=elements, num_speakers=speakers))


@app.post("/api/tag-object-stream", dependencies=[Depends(require_api_key)])
async def tag_object_stream(
    object_name: str = Form(..., alias="object"),
    filename: str = Form(default=""),
    top_k: int = Form(default=None, ge=1, le=527),
    threshold: float = Form(default=0.0, ge=0.0, le=1.0),
    segment_seconds: float = Form(default=0.0, ge=0.0, le=60.0),
    include_embedding: bool = Form(default=False),
    timeline_seconds: float = Form(default=0.0, ge=0.0, le=10.0),
    transcript_table: bool = Form(default=False),
    breakdown: bool = Form(default=False),
    script: Optional[UploadFile] = File(default=None),
    speakers: Optional[int] = Form(default=None, ge=1, le=12),
):
    """Same as /api/tag-object, but streams progress events while it works."""
    if tagger is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    elements = await _read_script(script)
    tmp_path, size = _fetch_object(object_name)
    t0 = time.time()
    k = top_k or settings.default_top_k
    name = filename or os.path.basename(object_name)
    return _analysis_stream(lambda progress: _tag_local_file(
        tmp_path, name, size, k, threshold, segment_seconds, include_embedding,
        timeline_seconds, transcript_table, breakdown, t0, progress,
        script_elements=elements, num_speakers=speakers))


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
