"""Interpretation column for the breakdown table, via Gemini.

Everything else in this service is measured from the audio. This module is the
one place that *infers*: it reads the detected rows and writes a short
plain-language reading of each. Its output is labelled as inference in the API
and the UI so it is never confused with what was actually detected.

Two auth modes, in order of preference:
  * Vertex AI (default on GCP): uses the runtime service account, no API key.
  * API key: set GEMINI_API_KEY.
"""
import json
import logging
import os
from typing import List, Optional

log = logging.getLogger("audiotagging.interpreter")

# Verified against the project: the 3.x flash models resolve only in the
# "global" location, not regional endpoints such as us-central1.
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

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

RESPONSE_SCHEMA = {
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
            },
        }
    },
    "required": ["rows"],
}


class Interpreter:
    """One structured-output call over the whole table."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL,
                 project: Optional[str] = None, location: str = "global",
                 max_rows: int = 300):
        from google import genai

        self.model = model
        self.max_rows = max_rows
        if api_key:
            self.client = genai.Client(api_key=api_key)
            self.mode = "api_key"
        else:
            # Application Default Credentials on Cloud Run - the runtime
            # service account needs roles/aiplatform.user.
            self.client = genai.Client(vertexai=True, project=project, location=location)
            self.mode = "vertex"

    def annotate(self, rows: List[dict]) -> List[str]:
        """One interpretation string per row (empty strings if the call fails)."""
        if not rows:
            return []
        from google.genai import types

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
            response = self.client.models.generate_content(
                model=self.model,
                contents="Rows:\n" + json.dumps(compact, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM,
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                    temperature=0.2,
                ),
            )
            payload = json.loads(response.text)
            by_index = {int(item["index"]): item.get("interpretation", "")
                        for item in payload.get("rows", [])}
        except Exception:
            log.exception("Interpretation call failed (model=%s, mode=%s)", self.model, self.mode)
            return [""] * len(rows)
        return [by_index.get(i, "") for i in range(len(rows))]
