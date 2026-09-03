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
        sed_model_type: Optional[str] = None,
        sed_checkpoint_path: Optional[str] = None,
    ):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.sample_rate = sample_rate
        self.hop_size = hop_size
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

        # Optional frame-level sound event detection model: gives a
        # probability per class roughly every 10 ms instead of one score per
        # clip, which is what lets /api/tag build a real timestamped timeline.
        self.sed_model = None
        if sed_model_type and sed_checkpoint_path and os.path.exists(sed_checkpoint_path):
            sed_cls = getattr(panns_models, sed_model_type)
            self.sed_model = sed_cls(
                sample_rate=sample_rate, window_size=window_size, hop_size=hop_size,
                mel_bins=mel_bins, fmin=fmin, fmax=fmax, classes_num=self.classes_num,
            )
            sed_ckpt = torch.load(sed_checkpoint_path, map_location="cpu", weights_only=False)
            self.sed_model.load_state_dict(sed_ckpt["model"] if "model" in sed_ckpt else sed_ckpt)
            self.sed_model.to(self.device)
            self.sed_model.eval()

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

    @torch.no_grad()
    def _forward_framewise(self, batch: np.ndarray) -> np.ndarray:
        """(batch, frames, classes_num) frame-level probabilities from the SED model."""
        x = torch.from_numpy(batch).to(self.device)
        out = self.sed_model(x, None)
        return out["framewise_output"].cpu().numpy()

    def detect_events(
        self,
        waveform: np.ndarray,
        bin_seconds: float = 1.0,
        top_k: int = 5,
        threshold: float = 0.15,
        merge_gap_seconds: float = 0.5,
        min_event_seconds: float = 0.15,
        max_events: int = 200,
    ) -> Optional[dict]:
        """Timestamped tags at roughly 10 ms resolution using the frame-level
        (sound event detection) model, if one was loaded.

        Returns a fixed-size ``timeline`` of ``bin_seconds`` windows (top tags
        per window, useful for scrubbing through the file) plus discrete
        ``events``: per-label runs where the frame probability stays above
        ``threshold``, merged across gaps shorter than ``merge_gap_seconds``
        and dropped if shorter than ``min_event_seconds``. Events still only
        name AudioSet classes (e.g. "Slam", "Thud", "Music") with a timestamp,
        not a description of what caused the sound.
        """
        if self.sed_model is None:
            return None

        duration = waveform.shape[0] / self.sample_rate
        framewise = self._forward_framewise(waveform[None, :])[0]  # (frames, classes)
        frames_per_second = self.sample_rate / self.hop_size
        n_frames = framewise.shape[0]

        bin_frames = max(1, int(round(bin_seconds * frames_per_second)))
        n_bins = int(np.ceil(n_frames / bin_frames))
        timeline = []
        for b in range(n_bins):
            start_f, end_f = b * bin_frames, min((b + 1) * bin_frames, n_frames)
            window_probs = framewise[start_f:end_f].max(axis=0)
            timeline.append({
                "start": round(start_f / frames_per_second, 2),
                "end": round(min(end_f / frames_per_second, duration), 2),
                "tags": self._top_k(window_probs, top_k, threshold),
            })

        gap_frames = max(1, int(round(merge_gap_seconds * frames_per_second)))
        above = framewise >= threshold
        events = []
        for class_idx in range(framewise.shape[1]):
            hits = np.flatnonzero(above[:, class_idx])
            if hits.size == 0:
                continue
            run_start = prev = int(hits[0])
            runs = []
            for i in hits[1:]:
                i = int(i)
                if i - prev <= gap_frames:
                    prev = i
                else:
                    runs.append((run_start, prev))
                    run_start = prev = i
            runs.append((run_start, prev))
            for start_f, end_f in runs:
                if (end_f - start_f + 1) / frames_per_second < min_event_seconds:
                    continue
                segment = framewise[start_f : end_f + 1, class_idx]
                peak_f = start_f + int(np.argmax(segment))
                # start_f/end_f/peak_f are now plain Python ints, so every
                # derived value below is a plain float, not numpy.float64
                # (which json.dumps cannot serialize).
                events.append({
                    "label": self.labels[class_idx],
                    "index": class_idx,
                    "start": round(start_f / frames_per_second, 2),
                    "end": round(min((end_f + 1) / frames_per_second, duration), 2),
                    "peak_time": round(peak_f / frames_per_second, 2),
                    "peak_probability": round(float(framewise[peak_f, class_idx]), 4),
                })

        if len(events) > max_events:
            events = sorted(events, key=lambda e: -e["peak_probability"])[:max_events]
        events.sort(key=lambda e: e["start"])

        return {
            "frames_per_second": round(frames_per_second, 2),
            "timeline": timeline,
            "events": events,
        }

    def tag(
        self,
        waveform: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0,
        segment_seconds: float = 0.0,
        include_embedding: bool = False,
        whole_clip_max_seconds: float = 60.0,
    ) -> dict:
        """Run clip-level tagging, optionally also per fixed-length segment.

        Clips up to ``whole_clip_max_seconds`` are fed to the model in one
        pass. Longer audio is processed in fixed windows (``segment_seconds``,
        default 10 s) and the clip-level result is the mean of the window
        probabilities, which keeps memory bounded for long recordings.
        """
        duration = waveform.shape[0] / self.sample_rate
        result = {
            "duration_seconds": round(duration, 3),
            "sample_rate": self.sample_rate,
            "model": self.model_type,
        }

        long_audio = duration > whole_clip_max_seconds
        if long_audio and not segment_seconds:
            segment_seconds = 10.0

        if not long_audio:
            clip_out = self._forward(waveform[None, :])
            clip_probs = clip_out["clipwise_output"][0]
            result["tags"] = self._top_k(clip_probs, top_k, threshold)
            result["aggregation"] = "whole_clip"
            if include_embedding and "embedding" in clip_out:
                result["embedding"] = clip_out["embedding"][0].round(5).tolist()

        if segment_seconds and duration > segment_seconds:
            seg_len = int(segment_seconds * self.sample_rate)
            n_segments = int(np.ceil(waveform.shape[0] / seg_len))
            padded = np.zeros(n_segments * seg_len, dtype=np.float32)
            padded[: waveform.shape[0]] = waveform
            batch = padded.reshape(n_segments, seg_len)
            segments = []
            all_probs = []
            embeddings = []
            batch_size = 8
            for start in range(0, n_segments, batch_size):
                out = self._forward(batch[start : start + batch_size])
                probs = out["clipwise_output"]
                all_probs.append(probs)
                if include_embedding and long_audio and "embedding" in out:
                    embeddings.append(out["embedding"])
                for i, p in enumerate(probs):
                    seg_idx = start + i
                    segments.append({
                        "start": round(seg_idx * segment_seconds, 3),
                        "end": round(min((seg_idx + 1) * segment_seconds, duration), 3),
                        "tags": self._top_k(p, top_k, threshold),
                    })
            result["segments"] = segments
            if long_audio:
                mean_probs = np.concatenate(all_probs, axis=0).mean(axis=0)
                result["tags"] = self._top_k(mean_probs, top_k, threshold)
                result["aggregation"] = "mean_over_segments"
                if embeddings:
                    result["embedding"] = np.concatenate(embeddings, 0).mean(0).round(5).tolist()
        return result
