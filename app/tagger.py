"""Model loading and inference helpers for the audio tagging web service.

Wraps the PANNs models in ``pytorch/models.py`` so they can be loaded once at
process start and reused across HTTP requests.
"""
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

import numpy as np
import soundfile
import torch

log = logging.getLogger("audiotagging.tagger")

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

    def framewise(self, waveform: np.ndarray, cache: Optional[dict] = None) -> Optional[np.ndarray]:
        """Frame-level probabilities, computed once and reused.

        The timeline and the breakdown table both need this pass; without the
        cache a request asking for both pays for the whole file twice.
        """
        if self.sed_model is None:
            return None
        if cache is not None and "framewise" in cache:
            return cache["framewise"]
        result = self._forward_framewise(waveform[None, :])[0]
        if cache is not None:
            cache["framewise"] = result
        return result

    def _label_group_indices(self, substrings, exact=frozenset()) -> List[int]:
        subs = [x.lower() for x in substrings]
        return [i for i, label in enumerate(self.labels)
                if any(x in label.lower() for x in subs) or label in exact]

    @staticmethod
    def _merge_runs(mask: np.ndarray, frames_per_second: float, merge_gap_seconds: float,
                    min_run_seconds: float) -> List[tuple]:
        """Contiguous (start_frame, end_frame) runs of True, merged across short gaps."""
        hits = np.flatnonzero(mask)
        if hits.size == 0:
            return []
        gap_frames = max(1, int(round(merge_gap_seconds * frames_per_second)))
        runs = []
        run_start = prev = int(hits[0])
        for i in hits[1:]:
            i = int(i)
            if i - prev <= gap_frames:
                prev = i
            else:
                runs.append((run_start, prev))
                run_start = prev = i
        runs.append((run_start, prev))
        min_frames = min_run_seconds * frames_per_second
        return [(a, b) for a, b in runs if (b - a + 1) >= min_frames]

    def detect_events(
        self,
        waveform: np.ndarray,
        bin_seconds: float = 1.0,
        top_k: int = 5,
        threshold: float = 0.15,
        merge_gap_seconds: float = 0.5,
        min_event_seconds: float = 0.15,
        max_events: int = 200,
        cache: Optional[dict] = None,
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
        framewise = self.framewise(waveform, cache)  # (frames, classes)
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

        above = framewise >= threshold
        events = []
        for class_idx in range(framewise.shape[1]):
            for start_f, end_f in self._merge_runs(above[:, class_idx], frames_per_second,
                                                   merge_gap_seconds, min_event_seconds):
                segment = framewise[start_f : end_f + 1, class_idx]
                peak_f = start_f + int(np.argmax(segment))
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

    def build_transcript_table(
        self,
        waveform: np.ndarray,
        transcriber,
        threshold: float = 0.15,
        merge_gap_seconds: float = 0.6,
        min_turn_seconds: float = 0.4,
        other_top_k: int = 2,
        max_rows: int = 400,
        cache: Optional[dict] = None,
    ) -> Optional[List[dict]]:
        """One row per continuous stretch of speech: when it ran, what was
        said, and what else (music, other sounds) was audible during it.

        Speech stretches come from the frame-level model; the text comes from
        the transcriber, which only ever sees those stretches, not the whole
        file. Non-speech columns carry labels and confidences, since only
        speech has words to transcribe.
        """
        if self.sed_model is None or transcriber is None:
            return None

        duration = waveform.shape[0] / self.sample_rate
        framewise = self.framewise(waveform, cache)
        frames_per_second = self.sample_rate / self.hop_size

        speech_idx = self._label_group_indices(
            ["speech"], exact={"Conversation", "Narration, monologue", "Babbling", "Whispering"})
        music_idx = self._label_group_indices(["music", "singing"])
        if not speech_idx:
            return []

        speech_mask = framewise[:, speech_idx].max(axis=1) >= threshold
        runs = self._merge_runs(speech_mask, frames_per_second, merge_gap_seconds, min_turn_seconds)[:max_rows]

        excluded = set(speech_idx) | set(music_idx)
        rows = []
        for start_f, end_f in runs:
            start_t = start_f / frames_per_second
            end_t = min((end_f + 1) / frames_per_second, duration)
            clip = waveform[int(start_t * self.sample_rate) : int(end_t * self.sample_rate)]
            window = framewise[start_f : end_f + 1]

            music = None
            if music_idx:
                music_peaks = window[:, music_idx].max(axis=0)
                best = int(np.argmax(music_peaks))
                if float(music_peaks[best]) >= threshold:
                    music = {"label": self.labels[music_idx[best]],
                             "probability": round(float(music_peaks[best]), 3)}

            other_probs = window.max(axis=0).copy()
            other_probs[list(excluded)] = 0.0

            rows.append({
                "start": round(start_t, 2),
                "end": round(end_t, 2),
                "speech": transcriber.transcribe(clip, self.sample_rate),
                "speech_confidence": round(float(window[:, speech_idx].max()), 3),
                "music": music,
                "other_sounds": self._top_k(other_probs, other_top_k, threshold),
            })
        return rows

    # Sound classes that imply a physical source, used to fill the Entity column
    # for event rows. Deliberately small and conservative: a guess here is worse
    # than a blank, since the column is meant to read as detected fact.
    ENTITY_HINTS = {
        "Knock": "Door", "Door": "Door", "Doorbell": "Door", "Slam": "Door",
        "Squeak": "Door/Hinge", "Sliding door": "Door", "Creak": "Door/Floor",
        "Footsteps": "Person", "Walk, footsteps": "Person", "Clapping": "Person",
        "Breathing": "Person", "Cough": "Person", "Sneeze": "Person",
        "Laughter": "Person", "Gasp": "Person", "Cheering": "Crowd",
        "Telephone": "Phone", "Telephone bell ringing": "Phone", "Ringtone": "Phone",
        "Dishes, pots, and pans": "Kitchenware", "Cutlery, silverware": "Kitchenware",
        "Water tap, faucet": "Tap", "Toilet flush": "Toilet", "Water": "Water",
        "Car": "Vehicle", "Vehicle": "Vehicle", "Engine": "Vehicle",
        "Car passing by": "Vehicle", "Bicycle": "Vehicle", "Train": "Vehicle",
        "Glass": "Glass", "Shatter": "Glass", "Chink, clink": "Glass",
        "Keys jangling": "Keys", "Typing": "Keyboard", "Computer keyboard": "Keyboard",
        "Dog": "Dog", "Cat": "Cat", "Bird": "Bird",
        "Wind": "Wind", "Rain": "Rain", "Thunder": "Weather",
    }

    def _delivery_metrics(self, clip: np.ndarray, text: str, duration: float,
                           pause_before: float) -> dict:
        """Loudness, pitch and pace, measured from the waveform."""
        metrics = {"pause_before": max(0.0, pause_before)}
        if clip.size:
            rms = float(np.sqrt(np.mean(np.square(clip, dtype=np.float64))))
            metrics["loudness_dbfs"] = round(20 * np.log10(max(rms, 1e-9)), 1)
        words = len([w for w in text.split() if w.strip()])
        metrics["words"] = words
        if duration > 0 and words:
            metrics["words_per_minute"] = round(words / duration * 60, 1)
        try:
            import librosa

            if clip.size >= self.sample_rate // 8:
                f0 = librosa.yin(np.asarray(clip, dtype=np.float32), fmin=65, fmax=400,
                                 sr=self.sample_rate)
                voiced = f0[np.isfinite(f0)]
                if voiced.size:
                    metrics["pitch_hz_median"] = round(float(np.median(voiced)), 1)
                    metrics["pitch_hz_range"] = round(
                        float(np.percentile(voiced, 90) - np.percentile(voiced, 10)), 1)
        except Exception:
            pass
        return metrics

    def build_breakdown_table(
        self,
        waveform: np.ndarray,
        transcriber=None,
        diarizer=None,
        threshold: float = 0.2,
        event_threshold: float = 0.35,
        merge_gap_seconds: float = 0.6,
        min_turn_seconds: float = 0.4,
        min_event_seconds: float = 0.15,
        max_rows: int = 300,
        cache: Optional[dict] = None,
        progress=None,
        num_speakers: Optional[int] = None,
    ) -> Optional[List[dict]]:
        """A production-style breakdown: one row per speech turn *and* per
        notable sound event, sorted by time.

        Speech rows carry the transcript (and a speaker label when a diarizer is
        supplied); event rows carry the detected sound and a conservative entity
        guess. Columns the audio cannot support (action, performance, camera)
        are returned empty for a human or a later video stage to fill.
        """
        if self.sed_model is None:
            return None

        duration = waveform.shape[0] / self.sample_rate
        framewise = self.framewise(waveform, cache)
        frames_per_second = self.sample_rate / self.hop_size

        speech_idx = self._label_group_indices(
            ["speech"], exact={"Conversation", "Narration, monologue", "Babbling", "Whispering"})
        music_idx = self._label_group_indices(["music", "singing"])
        spoken_or_musical = set(speech_idx) | set(music_idx)

        # --- speech turns -------------------------------------------------
        speech_mask = framewise[:, speech_idx].max(axis=1) >= threshold if speech_idx else \
            np.zeros(framewise.shape[0], dtype=bool)
        turns = self._merge_runs(speech_mask, frames_per_second, merge_gap_seconds, min_turn_seconds)

        spans = [(start_f / frames_per_second, min((end_f + 1) / frames_per_second, duration))
                 for start_f, end_f in turns]
        clips = [(waveform[int(a * self.sample_rate): int(b * self.sample_rate)], self.sample_rate)
                 for a, b in spans]

        if progress:
            progress("diarizing", 0.0)
        speakers = [None] * len(turns)
        if diarizer is not None and clips:
            try:
                speakers = diarizer.label_turns(clips, num_speakers=num_speakers)
            except Exception:
                speakers = [None] * len(turns)

        if progress:
            progress("transcribing", 0.0)
        dialogue = [""] * len(turns)
        if transcriber is not None and spans:
            try:
                dialogue = transcriber.transcribe_turns(waveform, self.sample_rate, spans)
            except Exception:
                log.exception("Transcription failed")

        rows = []
        previous_end = 0.0
        for (start_f, end_f), (start_t, end_t), (clip, _), speaker, text in zip(
                turns, spans, clips, speakers, dialogue):
            window = framewise[start_f : end_f + 1]
            concurrent = window.max(axis=0).copy()
            concurrent[list(spoken_or_musical)] = 0.0
            music_present = bool(music_idx) and float(window[:, music_idx].max()) >= threshold
            rows.append({
                "kind": "speech",
                "start": round(start_t, 2),
                "end": round(end_t, 2),
                "dialogue": text,
                "audio": "Music" if music_present else "",
                "entity": speaker or "",
                "confidence": round(float(window[:, speech_idx].max()), 3) if speech_idx else None,
                "other_sounds": self._top_k(concurrent, 2, event_threshold),
                # Measured delivery cues - useful as direction input, and unlike
                # the interpretation column these are read off the waveform.
                "measured": self._delivery_metrics(clip, text, end_t - start_t,
                                                   round(start_t - previous_end, 2)),
            })
            previous_end = end_t

        # --- sound events (non-speech, non-music) -------------------------
        above = framewise >= event_threshold
        for class_idx in range(framewise.shape[1]):
            if class_idx in spoken_or_musical:
                continue
            for start_f, end_f in self._merge_runs(above[:, class_idx], frames_per_second,
                                                   merge_gap_seconds, min_event_seconds):
                segment = framewise[start_f : end_f + 1, class_idx]
                peak_f = start_f + int(np.argmax(segment))
                label = self.labels[class_idx]
                rows.append({
                    "kind": "event",
                    "start": round(start_f / frames_per_second, 2),
                    "end": round(min((end_f + 1) / frames_per_second, duration), 2),
                    "dialogue": "",
                    "audio": label,
                    "entity": self.ENTITY_HINTS.get(label, ""),
                    "confidence": round(float(framewise[peak_f, class_idx]), 3),
                    "other_sounds": [],
                })

        # Keep the most confident rows if there are too many, but always keep
        # every speech row - the dialogue is the point of the table.
        if len(rows) > max_rows:
            speech_rows = [r for r in rows if r["kind"] == "speech"]
            event_rows = sorted((r for r in rows if r["kind"] == "event"),
                                key=lambda r: -(r["confidence"] or 0))
            rows = speech_rows + event_rows[: max(0, max_rows - len(speech_rows))]
        rows.sort(key=lambda r: (r["start"], r["kind"] != "speech"))

        if progress:
            progress("assembling", 0.9)
        for row in rows:
            row.setdefault("measured", {})
            row["interpretation"] = ""
            row["action"] = ""
            row["performance"] = ""
            row["camera"] = ""
        return rows

    # Words in stage directions -> AudioSet label fragments that would confirm
    # them, plus the physical source for the Entity column. Matched against the
    # detected events inside the direction's time window.
    DIRECTION_CUES = [
        (("knock", "thump", "hammer", "bang", "pound"), ("knock", "thump", "bang", "slam", "thud"), "Door"),
        (("door",), ("door", "slam", "knock"), "Door"),
        (("quack", "duck", "squeak"), ("squeak", "quack", "duck"), "Rubber duck"),
        (("water", "sink", "tap", "splash", "shower"), ("water", "tap", "pour", "drip", "splash", "liquid", "gurgl"), "Water"),
        (("music", "party", "atmosphere"), ("music",), "Party"),
        (("curtain", "whisk", "snatch"), ("whoosh", "swish", "rustle", "fabric"), "Shower curtain"),
        (("silence", "quiet", "peace"), ("silence",), ""),
        (("giggle", "laugh"), ("laugh", "giggle", "chuckle"), ""),
        (("hiccup",), ("hiccup",), ""),
        (("shout", "scream", "yell"), ("shout", "yell", "scream"), ""),
        (("footstep", "walk", "stagger", "run"), ("footstep", "walk", "run"), ""),
        (("slump", "sit", "fall", "collapse"), ("thump", "thud"), ""),
        (("glass", "bottle"), ("glass", "clink", "shatter"), "Glass"),
        (("phone", "ring"), ("telephone", "ring"), "Phone"),
    ]

    def _direction_cue(self, text: str):
        low = text.lower()
        for triggers, label_fragments, entity in self.DIRECTION_CUES:
            if any(t in low for t in triggers):
                return label_fragments, entity
        return (), ""

    def build_script_breakdown(
        self,
        waveform: np.ndarray,
        elements,
        transcriber,
        threshold: float = 0.2,
        event_threshold: float = 0.3,
        merge_gap_seconds: float = 0.6,
        min_turn_seconds: float = 0.4,
        cache: Optional[dict] = None,
        progress=None,
    ) -> Optional[dict]:
        """Timestamp a script against the recording.

        The script is authoritative for who speaks and what they say; the audio
        supplies *when*. Dialogue lines are placed by aligning script words to
        Whisper word timestamps; stage directions are placed in the gap between
        their neighbouring lines and snapped to a matching detected sound when
        one exists there (a "THUMPING AT THE DOOR" direction snaps to the
        detected Knock). Rows keep script order.
        """
        if self.sed_model is None or transcriber is None:
            return None
        from .aligner import place_lines
        from .script_parser import characters, proper_nouns

        duration = waveform.shape[0] / self.sample_rate
        framewise = self.framewise(waveform, cache)
        fps = self.sample_rate / self.hop_size

        # 1. speech spans -> word timestamps (with script vocabulary as hints)
        speech_idx = self._label_group_indices(
            ["speech"], exact={"Conversation", "Narration, monologue", "Babbling", "Whispering"})
        speech_mask = framewise[:, speech_idx].max(axis=1) >= threshold
        spans = [(a / fps, min((b + 1) / fps, duration))
                 for a, b in self._merge_runs(speech_mask, fps, merge_gap_seconds, min_turn_seconds)]
        if progress:
            progress("transcribing", 0.0)
        words = transcriber.transcribe_words(waveform, self.sample_rate, spans,
                                             hotwords=proper_nouns(elements))

        # 2. place dialogue lines
        if progress:
            progress("aligning script", 0.6)
        line_elements = [e for e in elements if e.kind == "line"]
        placed = place_lines([e.text for e in line_elements], words)
        timing = {e.index: pl for e, pl in zip(line_elements, placed)}

        # 3. detected events, for snapping directions and filling Audio
        music_idx = set(self._label_group_indices(["music", "singing"]))
        skip = set(speech_idx) | music_idx
        events = []
        above = framewise >= event_threshold
        for ci in range(framewise.shape[1]):
            if ci in skip:
                continue
            for a, b in self._merge_runs(above[:, ci], fps, 0.5, 0.1):
                peak = a + int(np.argmax(framewise[a:b + 1, ci]))
                events.append({"label": self.labels[ci], "start": a / fps, "end": (b + 1) / fps,
                               "peak_time": peak / fps, "prob": float(framewise[peak, ci])})
        events.sort(key=lambda e: e["start"])

        def events_between(t0: float, t1: float):
            return [e for e in events if e["end"] >= t0 and e["start"] <= t1]

        # 4. rows in script order
        rows, scene = [], None
        n = len(elements)
        for i, el in enumerate(elements):
            if el.kind == "scene":
                scene = el.text
                continue
            if el.kind == "line":
                pl = timing[el.index]
                a_f, b_f = int(pl["start"] * fps), max(int(pl["start"] * fps) + 1, int(pl["end"] * fps))
                window = framewise[a_f:b_f] if b_f > a_f and a_f < framewise.shape[0] else framewise[:1]
                music_present = bool(music_idx) and float(window[:, sorted(music_idx)].max()) >= threshold
                clip = waveform[int(pl["start"] * self.sample_rate): int(pl["end"] * self.sample_rate)]
                concurrent = [e["label"] for e in events_between(pl["start"], pl["end"])
                              if e["prob"] >= 0.4][:2]
                rows.append({
                    "kind": "line",
                    "start": pl["start"], "end": pl["end"],
                    "dialogue": el.text,
                    "heard": pl["heard"],
                    "alignment": pl["coverage"],
                    "estimated": pl["estimated"],
                    "audio": ", ".join((["Music"] if music_present else []) + concurrent),
                    "entity": el.character.title() if el.character else "",
                    "notes": el.notes,                       # authored performance notes
                    "measured": self._delivery_metrics(clip, el.text, max(0.0, pl["end"] - pl["start"]), 0.0)
                    if clip.size else {},
                })
                continue

            # direction: window between neighbouring placed lines
            prev_end = next((timing[elements[k].index]["end"] for k in range(i - 1, -1, -1)
                             if elements[k].kind == "line"), 0.0)
            next_start = next((timing[elements[k].index]["start"] for k in range(i + 1, n)
                               if elements[k].kind == "line"), duration)
            if next_start < prev_end:
                next_start = prev_end
            fragments, entity = self._direction_cue(el.text)
            lo, hi = prev_end - 0.5, next_start + 0.5
            candidates = events_between(lo, hi)
            match = None
            if fragments:
                hits = [e for e in candidates if any(f in e["label"].lower() for f in fragments)]
                if hits:
                    # prefer a sound that peaks inside this gap; the direction
                    # belongs here even if the detection also runs on past it
                    inside = [e for e in hits if lo <= e["peak_time"] <= hi]
                    match = max(inside or hits, key=lambda e: e["prob"])
            if match:
                # clip the detection to the window so a long-running sound
                # (music, a tap left on) cannot stretch the direction over
                # the lines around it
                m_start = max(match["start"], prev_end)
                m_end = min(match["end"], next_start)
                if m_end <= m_start:
                    m_start = min(max(match["peak_time"], prev_end), next_start)
                    m_end = m_start
            rows.append({
                "kind": "direction",
                "start": round(m_start if match else prev_end, 2),
                "end": round(m_end if match else next_start, 2),
                "dialogue": "",
                "heard": "",
                "alignment": None,
                "estimated": match is None,
                "action": el.text,                            # authored
                "audio": (f'{match["label"]} ({match["prob"]:.2f})' if match else
                          ", ".join(f'{e["label"]}' for e in sorted(candidates, key=lambda e: -e["prob"])[:2])),
                "entity": entity,
                "notes": [],
                "measured": {},
            })

        # pause before each line, now that everything is placed
        last_end = 0.0
        for r in rows:
            if r["kind"] == "line":
                r["measured"]["pause_before"] = round(max(0.0, r["start"] - last_end), 2)
            last_end = max(last_end, r["end"])

        return {
            "scene": scene,
            "characters": [c.title() for c in characters(elements)],
            "rows": rows,
            "words_recognised": len(words),
            "lines_placed": sum(1 for r in rows if r["kind"] == "line" and not r["estimated"]),
            "lines_total": sum(1 for r in rows if r["kind"] == "line"),
        }
