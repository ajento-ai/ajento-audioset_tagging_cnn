"""Model loading and inference helpers for the audio tagging web service.

Wraps the PANNs models in ``pytorch/models.py`` so they can be loaded once at
process start and reused across HTTP requests.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

import numpy as np
import soundfile
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "pytorch"))
sys.path.insert(0, os.path.join(REPO_ROOT, "utils"))

import models as panns_models  # noqa: E402


def load_labels(csv_path: str) -> List[str]:
    import csv

    with open(csv_path, "r") as f:
        rows = list(csv.reader(f, delimiter=","))
    return [row[2] for row in rows[1:]]


class AudioTagger:
    """Holds a loaded PANNs model plus its feature-extraction settings."""

    def __init__(
        self,
        model_type: str = "Cnn14",
        checkpoint_path: Optional[str] = None,
        sample_rate: int = 32000,
        window_size: int = 1024,
        hop_size: int = 320,
        mel_bins: int = 64,
        fmin: int = 50,
        fmax: int = 14000,
        labels_csv: Optional[str] = None,
        device: Optional[str] = None,
        num_threads: Optional[int] = None,
    ):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.sample_rate = sample_rate
        labels_csv = labels_csv or os.path.join(
            REPO_ROOT, "metadata", "class_labels_indices.csv"
        )
        self.labels = load_labels(labels_csv)
        self.classes_num = len(self.labels)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if num_threads:
            torch.set_num_threads(num_threads)

        model_cls = getattr(panns_models, model_type)
        self.model = model_cls(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=self.classes_num,
        )
        if checkpoint_path:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            state = checkpoint["model"] if "model" in checkpoint else checkpoint
            self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------ audio
    def decode_audio(self, path: str, max_seconds: Optional[float] = None) -> np.ndarray:
        """Decode any ffmpeg-readable file to mono float32 at the model rate."""
        if shutil.which("ffmpeg"):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name
            try:
                cmd = [
                    "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                    "-i", path,
                ]
                if max_seconds:
                    cmd += ["-t", str(max_seconds)]
                cmd += [
                    "-ac", "1", "-ar", str(self.sample_rate),
                    "-f", "wav", "-acodec", "pcm_f32le", wav_path,
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    raise ValueError(
                        "ffmpeg could not decode the file: "
                        + proc.stderr.strip()[-500:]
                    )
                waveform, _ = soundfile.read(wav_path, dtype="float32", always_2d=False)
            finally:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
        else:
            import librosa

            try:
                waveform, _ = librosa.load(
                    path, sr=self.sample_rate, mono=True,
                    duration=max_seconds if max_seconds else None,
                )
            except Exception as e:  # librosa/soundfile/audioread raise many types
                raise ValueError(f"Could not decode audio file: {e}") from e
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if waveform.size == 0:
            raise ValueError("Decoded audio is empty")
        return waveform

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def _forward(self, batch: np.ndarray) -> Dict[str, np.ndarray]:
        x = torch.from_numpy(batch).to(self.device)
        out = self.model(x, None)
        result = {"clipwise_output": out["clipwise_output"].cpu().numpy()}
        if "embedding" in out:
            result["embedding"] = out["embedding"].cpu().numpy()
        return result

    def _top_k(self, probs: np.ndarray, top_k: int, threshold: float) -> List[dict]:
        order = np.argsort(probs)[::-1]
        tags = []
        for idx in order[:top_k]:
            p = float(probs[idx])
            if p < threshold:
                break
            tags.append({"index": int(idx), "label": self.labels[idx], "probability": round(p, 4)})
        return tags

    def tag(
        self,
        waveform: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0,
        segment_seconds: float = 0.0,
        include_embedding: bool = False,
    ) -> dict:
        """Run clip-level tagging, optionally also per fixed-length segment."""
        duration = waveform.shape[0] / self.sample_rate
        clip_out = self._forward(waveform[None, :])
        clip_probs = clip_out["clipwise_output"][0]

        result = {
            "duration_seconds": round(duration, 3),
            "sample_rate": self.sample_rate,
            "model": self.model_type,
            "tags": self._top_k(clip_probs, top_k, threshold),
        }
        if include_embedding and "embedding" in clip_out:
            result["embedding"] = clip_out["embedding"][0].round(5).tolist()

        if segment_seconds and duration > segment_seconds:
            seg_len = int(segment_seconds * self.sample_rate)
            n_segments = int(np.ceil(waveform.shape[0] / seg_len))
            padded = np.zeros(n_segments * seg_len, dtype=np.float32)
            padded[: waveform.shape[0]] = waveform
            batch = padded.reshape(n_segments, seg_len)
            segments = []
            batch_size = 8
            for start in range(0, n_segments, batch_size):
                probs = self._forward(batch[start : start + batch_size])["clipwise_output"]
                for i, p in enumerate(probs):
                    seg_idx = start + i
                    segments.append({
                        "start": round(seg_idx * segment_seconds, 3),
                        "end": round(min((seg_idx + 1) * segment_seconds, duration), 3),
                        "tags": self._top_k(p, top_k, threshold),
                    })
            result["segments"] = segments
        return result
