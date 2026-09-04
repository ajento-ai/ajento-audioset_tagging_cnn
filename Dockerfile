# Audio tagging web service (PANNs Cnn14, CPU inference) for Cloud Run.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# CPU-only torch keeps the image ~1.5 GB smaller than the default CUDA wheel.
ARG TORCH_VERSION=2.4.1
RUN pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu
COPY app/requirements.txt app/requirements.txt
# webrtcvad (a Resemblyzer dependency) ships no wheel and needs a compiler.
# Install the toolchain, build, then purge it so the image stays lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc python3-dev \
    && pip install -r app/requirements.txt \
    && apt-get purge -y gcc python3-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Pretrained checkpoint (~330 MB). Baked into the image so cold starts do not
# depend on Zenodo. Override MODEL_URL / MODEL_FILE to ship another PANNs model.
ARG MODEL_URL="https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
ARG MODEL_FILE="Cnn14_mAP=0.431.pth"
RUN mkdir -p /models && curl -fSL --retry 5 --retry-delay 5 -o "/models/${MODEL_FILE}" "${MODEL_URL}"

# Frame-level (sound event detection) model, used for the timestamped
# timeline/events feature. Set SED_MODEL_URL="" to skip it and build without.
ARG SED_MODEL_URL="https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"
ARG SED_MODEL_FILE="Cnn14_DecisionLevelMax_mAP=0.385.pth"
RUN if [ -n "${SED_MODEL_URL}" ]; then \
      curl -fSL --retry 5 --retry-delay 5 -o "/models/${SED_MODEL_FILE}" "${SED_MODEL_URL}"; \
    fi

# Pre-download the Whisper weights so cold starts do not hit Hugging Face.
# Set WHISPER_MODEL_SIZE="" to build without speech-to-text.
ARG WHISPER_MODEL_SIZE=base
RUN if [ -n "${WHISPER_MODEL_SIZE}" ]; then \
      python3 -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL_SIZE}', device='cpu', compute_type='int8', download_root='/models/whisper')"; \
    fi

COPY metadata/class_labels_indices.csv metadata/class_labels_indices.csv
COPY pytorch/models.py pytorch/pytorch_utils.py pytorch/
COPY app app

ENV MODEL_TYPE=Cnn14 \
    CHECKPOINT_PATH=/models/${MODEL_FILE} \
    SED_MODEL_TYPE=Cnn14_DecisionLevelMax \
    SED_CHECKPOINT_PATH=/models/${SED_MODEL_FILE} \
    WHISPER_MODEL_SIZE=${WHISPER_MODEL_SIZE} \
    WHISPER_MODEL_DIR=/models/whisper \
    PORT=8080

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fs http://localhost:8080/healthz || exit 1

# Single worker: the model is ~330 MB in RAM; scale via Cloud Run instances instead.
CMD exec uvicorn app.server:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 75
