"""Interpretation column for the breakdown table, via the Claude API.

Everything else in this service is measured from the audio. This module is the
one place that *infers*: it reads the detected rows and writes a short
plain-language reading of each. Its output is labelled as inference in the API
and the UI so it is never confused with what was actually detected.
"""
import json
import logging
import os
from typing import List, Optional

log = logging.getLogger("audiotagging.interpreter")

MODEL = "claude-opus-5"

SYSTEM = """You annotate an audio breakdown for a film/production team.

You receive rows detected from an audio track. Each row is either a stretch of
speech (with a transcript) or a detected sound event, with a timestamp.

For each row, write one short interpretation (max 12 words) of what is likely
happening at that moment, using the surrounding rows for context. Examples:
"Knock from outside the room", "Jane reacts", "Music swells under dialogue".

Rules:
- Base it on the given rows only. Never invent dialogue, names, or locations
  that the rows do not support.
- If a row is ambiguous, say the plain reading ("Impact sound, source unclear")
  rather than guessing a specific cause.
- Refer to speakers exactly as the rows label them (e.g. "Speaker 1"), unless
  the transcript itself names someone.
- Return one entry per input row, in the same order, keyed by row index.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "interpretation": {"type": "string"},
                },
                "required": ["index", "interpretation"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rows"],
    "additionalProperties": False,
}


class Interpreter:
    """Wraps a single structured-output call over the whole table."""

    def __init__(self, api_key: Optional[str] = None, model: str = MODEL,
                 max_rows: int = 300):
        import anthropic

        self.model = model
        self.max_rows = max_rows
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def annotate(self, rows: List[dict]) -> List[str]:
        """Returns one interpretation string per row (empty string on failure)."""
        if not rows:
            return []
        compact = [
            {
                "index": i,
                "time": f'{r["start"]}-{r["end"]}s',
                "kind": r["kind"],
                "speaker": r.get("entity") or "",
                "dialogue": (r.get("dialogue") or "")[:300],
                "audio": r.get("audio") or "",
            }
            for i, r in enumerate(rows[: self.max_rows])
        ]
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM,
                messages=[{"role": "user",
                           "content": "Rows:\n" + json.dumps(compact, ensure_ascii=False)}],
                output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            )
            if response.stop_reason == "refusal":
                log.warning("Interpretation declined: %s", response.stop_details)
                return [""] * len(rows)
            text = next(b.text for b in response.content if b.type == "text")
            by_index = {int(item["index"]): item["interpretation"]
                        for item in json.loads(text)["rows"]}
        except Exception:
            log.exception("Interpretation call failed")
            return [""] * len(rows)
        return [by_index.get(i, "") for i in range(len(rows))]
