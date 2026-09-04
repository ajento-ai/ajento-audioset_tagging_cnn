"""Speaker labels for speech turns.

Embeds each speech turn with Resemblyzer's voice encoder (weights ship with the
package, so no gated model download) and clusters the embeddings, giving
"Speaker 1", "Speaker 2", ... A cast list can map those onto real names.
"""
import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("audiotagging.diarizer")

RESEMBLYZER_SAMPLE_RATE = 16000
MIN_SECONDS = 0.4


class SpeakerDiarizer:
    def __init__(self, max_speakers: int = 6, distance_threshold: float = 0.35):
        # 0.35 cosine distance: measured same-speaker embeddings sit well below
        # it and different speakers well above. Raising it merges speakers;
        # lowering it splits one speaker across turns.
        from resemblyzer import VoiceEncoder

        self.encoder = VoiceEncoder("cpu")
        self.max_speakers = max_speakers
        self.distance_threshold = distance_threshold

    def _embed(self, clip: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        if clip.size / sample_rate < MIN_SECONDS:
            return None
        if sample_rate != RESEMBLYZER_SAMPLE_RATE:
            import librosa

            clip = librosa.resample(np.asarray(clip, dtype=np.float32),
                                    orig_sr=sample_rate, target_sr=RESEMBLYZER_SAMPLE_RATE)
        try:
            return self.encoder.embed_utterance(np.asarray(clip, dtype=np.float32))
        except Exception:
            log.exception("Could not embed speech turn")
            return None

    def label_turns(self, clips: Sequence[Tuple[np.ndarray, int]]) -> List[Optional[str]]:
        """One "Speaker N" label per clip; None where the clip was too short."""
        embeddings, positions = [], []
        for i, (clip, sample_rate) in enumerate(clips):
            emb = self._embed(clip, sample_rate)
            if emb is not None:
                embeddings.append(emb)
                positions.append(i)

        labels: List[Optional[str]] = [None] * len(clips)
        if not embeddings:
            return labels
        if len(embeddings) == 1:
            labels[positions[0]] = "Speaker 1"
            return labels

        from sklearn.cluster import AgglomerativeClustering

        matrix = np.vstack(embeddings)
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=self.distance_threshold,
            metric="cosine", linkage="average",
        ).fit(matrix)
        assignments = clustering.labels_

        # Number speakers by first appearance so "Speaker 1" is whoever talks first.
        order, renumbered = {}, []
        for a in assignments:
            if a not in order:
                order[a] = len(order) + 1
            renumbered.append(order[a])
        for pos, speaker in zip(positions, renumbered):
            labels[pos] = f"Speaker {min(speaker, self.max_speakers)}"
        return labels
