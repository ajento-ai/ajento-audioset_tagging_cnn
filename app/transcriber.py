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

    def transcribe_turns(self, waveform: np.ndarray, sample_rate: int,
                         turns: "list[tuple[float, float]]",
                         window_seconds: float = 28.0) -> "list[str]":
        """Transcribe many speech turns with far fewer encoder passes.

        Whisper always encodes a 30-second window, so transcribing a 2-second
        line costs nearly as much as a 30-second one. Consecutive turns are
        packed into shared windows and the returned segments are assigned back
        to turns by time overlap - fewer passes, and Whisper also reads better
        with continuous audio than with clipped fragments.
        """
        if not turns:
            return []

        groups, current = [], [0]
        for i in range(1, len(turns)):
            span = turns[i][1] - turns[current[0]][0]
            if span <= window_seconds:
                current.append(i)
            else:
                groups.append(current)
                current = [i]
        groups.append(current)

        texts = [""] * len(turns)
        for group in groups:
            g_start, g_end = turns[group[0]][0], turns[group[-1]][1]
            clip = waveform[int(g_start * sample_rate): int(g_end * sample_rate)]
            for idx, seg_start, seg_end, text in self._segments(clip, sample_rate, g_start):
                # attach each recognised segment to the turn it overlaps most
                best, best_overlap = None, 0.0
                for i in group:
                    t_start, t_end = turns[i]
                    overlap = min(seg_end, t_end) - max(seg_start, t_start)
                    if overlap > best_overlap:
                        best, best_overlap = i, overlap
                if best is not None:
                    texts[best] = (texts[best] + " " + text).strip()
        return texts

    def _segments(self, clip: np.ndarray, sample_rate: int, offset: float):
        """Yield (index, absolute_start, absolute_end, text) for one window."""
        if clip.size == 0:
            return
        if sample_rate != WHISPER_SAMPLE_RATE:
            import librosa

            clip = librosa.resample(np.asarray(clip, dtype=np.float32),
                                    orig_sr=sample_rate, target_sr=WHISPER_SAMPLE_RATE)
        segments, _ = self.model.transcribe(np.asarray(clip, dtype=np.float32),
                                            beam_size=self.beam_size, vad_filter=False)
        for i, segment in enumerate(segments):
            text = segment.text.strip()
            if text:
                yield i, offset + segment.start, offset + segment.end, text

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
