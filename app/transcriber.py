"""Speech-to-text for the transcript table, via faster-whisper (CTranslate2).

Only used to fill the transcript column of segments the tagger already
identified as speech; it is never run over a whole file.
"""
import logging
from typing import Optional

import numpy as np

log = logging.getLogger("audiotagging.transcriber")

WHISPER_SAMPLE_RATE = 16000


class SpeechTranscriber:
    def __init__(self, model_size: str = "base", download_root: Optional[str] = None,
                 device: str = "cpu", compute_type: str = "int8", beam_size: int = 1):
        import os

        from faster_whisper import WhisperModel

        self.beam_size = beam_size
        self.model_size = model_size
        # The weights are baked into the image, but faster-whisper still calls
        # Hugging Face on startup to check the revision - which has already come
        # back 429 in production. Pin to the local copy when it is present so a
        # cold start never depends on that call.
        local_only = bool(download_root) and os.path.isdir(download_root) and bool(
            os.listdir(download_root))
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type,
                                      download_root=download_root, local_files_only=local_only)
        except Exception:
            if not local_only:
                raise
            log.warning("Local Whisper weights unusable; retrying with download")
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type,
                                      download_root=download_root)

    def transcribe(self, waveform: np.ndarray, sample_rate: int) -> str:
        if waveform.size == 0:
            return ""
        if sample_rate != WHISPER_SAMPLE_RATE:
            import librosa

            waveform = librosa.resample(np.asarray(waveform, dtype=np.float32),
                                        orig_sr=sample_rate, target_sr=WHISPER_SAMPLE_RATE)
        segments, _ = self.model.transcribe(np.asarray(waveform, dtype=np.float32),
                                            beam_size=self.beam_size, vad_filter=False)
        return " ".join(segment.text.strip() for segment in segments).strip()
