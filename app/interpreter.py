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

SYSTEM = """You turn a detected audio breakdown into shot direction for a team
generating video from this audio.

Each row is either a stretch of speech (with a transcript) or a detected sound
event, with a timestamp. Speech rows may carry measured delivery cues:
loudness in dBFS, median pitch and pitch range in Hz, words per minute, and the
pause before the line. Use those cues as evidence for how a line is delivered -
a loud, wide-pitch, fast line reads as agitated; a quiet, narrow, slow one after
a long pause reads as subdued.

For each row produce four short fields:
- interpretation: what is happening at this moment (max 12 words).
- action: the physical action or blocking implied (max 10 words), "" if the
  audio implies none.
- performance: how the line is delivered, grounded in the measured cues
  (max 8 words), "" for non-speech rows.
- camera: one shot suggestion that serves this beat (max 8 words), e.g.
  "close on reacting listener", "wide, hold on doorway".

Rules:
- interpretation must stay faithful to the rows: never invent dialogue, names
  or locations they do not support. If a row is ambiguous, give the plain
  reading ("Impact sound, source unclear") rather than guessing a cause.
- action, performance and camera are creative suggestions, not claims about the
  recording. Keep them consistent with neighbouring rows so the sequence reads
  as one scene.
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
                    "action": {"type": "string"},
                    "performance": {"type": "string"},
                    "camera": {"type": "string"},
                },
                "required": ["index", "interpretation", "action", "performance", "camera"],
            },
        }
    },
    "required": ["rows"],
}


SCRIPT_SYSTEM = """You turn a timestamped script into shot direction for a team
generating video from a recorded performance of it.

Each row is a script element placed in time against the recording:
- line rows: the character, the scripted dialogue, what Whisper actually heard
  (for spotting deviations), authored performance notes from the script such as
  "SHOUTING" or "BABY VOICE", and measured delivery cues (loudness dBFS, median
  pitch and range in Hz, words per minute, pause before the line).
- direction rows: the authored stage direction / sound cue, and the detected
  sound that matched it (or the sounds heard nearby if none matched).
The scene heading gives the location.

For each row produce:
- interpretation: what this beat is doing in the scene (max 12 words).
- action: for line rows only, the physical business implied for the speaker
  (max 10 words) consistent with the surrounding directions; "" if none.
  For direction rows return "" - the authored direction already is the action.
- performance: for line rows, how the line is delivered - start from the
  authored notes, then use the measured cues to say more (max 10 words). ""
  for direction rows.
- camera: one shot suggestion that serves this beat (max 8 words), consistent
  across neighbouring rows so the sequence cuts as one scene.

Rules: never contradict the script; use character names as given; if a heard
text clearly differs from the scripted line, mention it in interpretation
("delivered as 'X' - deviates from script"). Return one entry per row, keyed by
row index, same order.
"""


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

    FIELDS = ("interpretation", "action", "performance", "camera")

    def annotate(self, rows: List[dict], scene: Optional[str] = None,
                 script_mode: bool = False) -> List[dict]:
        """One {interpretation, action, performance, camera} dict per row."""
        blank = [{f: "" for f in self.FIELDS} for _ in rows]
        if not rows:
            return []
        from google.genai import types

        compact = []
        for i, r in enumerate(rows[: self.max_rows]):
            item = {
                "index": i,
                "time": f'{r["start"]}-{r["end"]}s',
                "kind": r["kind"],
                "speaker": r.get("entity") or "",
                "dialogue": (r.get("dialogue") or "")[:300],
                "audio": r.get("audio") or "",
                "measured": r.get("measured") or {},
            }
            if script_mode:
                item["heard"] = (r.get("heard") or "")[:300]
                item["authored_notes"] = r.get("notes") or []
                item["authored_direction"] = r.get("action") or ""
            compact.append(item)
        prompt = "Rows:\n" + json.dumps(compact, ensure_ascii=False)
        if scene:
            prompt = f"Scene: {scene}\n" + prompt
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SCRIPT_SYSTEM if script_mode else SYSTEM,
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                    temperature=0.2,
                ),
            )
            payload = json.loads(response.text)
            by_index = {
                int(item["index"]): {f: str(item.get(f, "") or "") for f in self.FIELDS}
                for item in payload.get("rows", [])
            }
        except Exception:
            log.exception("Interpretation call failed (model=%s, mode=%s)", self.model, self.mode)
            return blank
        return [by_index.get(i, {f: "" for f in self.FIELDS}) for i in range(len(rows))]
