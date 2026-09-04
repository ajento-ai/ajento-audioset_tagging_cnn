"""Parse a radio/screen script into ordered elements.

Convention (matching the scripts this service is used with): a line in ALL
CAPS is a stage direction or sound cue, unless it is a character cue
("JANE:"). Everything else is dialogue for the most recent character.
Parentheticals inside dialogue, "(SHOUTING)", are performance notes.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

CUE_RE = re.compile(r"^([A-Z][A-Z0-9 .'\-]{0,40}):\s*(.*)$")
SCENE_RE = re.compile(r"^(SCENE\s+\S+\.?\s*)?(INT\.|EXT\.|INT/EXT)", re.I)
PAREN_RE = re.compile(r"\(([^)]*)\)")
PAGE_RE = re.compile(r"^\s*page\s+\d+\s+of\s+\d+\s*$", re.I)
FX_PREFIX_RE = re.compile(r"^(FX|SFX|SOUND|MUSIC|GRAMS)\s*:\s*", re.I)


@dataclass
class Element:
    kind: str                     # "scene" | "direction" | "line"
    text: str                     # dialogue text (parentheticals removed) or direction text
    character: Optional[str] = None
    notes: List[str] = field(default_factory=list)   # parentheticals, e.g. SHOUTING
    index: int = 0


def _is_caps(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _clean_pdf_text(text: str) -> List[str]:
    """Drop running headers/footers and page numbers a PDF export leaves behind."""
    lines = [ln.strip() for ln in text.splitlines()]
    counts = {}
    for ln in lines:
        if ln:
            counts[ln] = counts.get(ln, 0) + 1
    out = []
    for ln in lines:
        if not ln or PAGE_RE.match(ln):
            continue
        # a non-dialogue line repeated on every page is a header
        if counts.get(ln, 0) >= 3 and not CUE_RE.match(ln) and len(ln) > 20 and not _is_caps(ln):
            continue
        out.append(ln)
    return out


def parse_script(text: str) -> List[Element]:
    lines = _clean_pdf_text(text)
    elements: List[Element] = []
    current_char: Optional[str] = None
    buffer: List[str] = []
    buffer_notes: List[str] = []

    def flush_line():
        nonlocal buffer, buffer_notes
        if current_char and (buffer or buffer_notes):
            raw = " ".join(buffer).strip()
            notes = [n.strip() for n in PAREN_RE.findall(raw) if n.strip()] + buffer_notes
            spoken = PAREN_RE.sub(" ", raw)
            spoken = re.sub(r"\s+", " ", spoken).strip()
            elements.append(Element("line", spoken, current_char, notes))
        buffer, buffer_notes = [], []

    for ln in lines:
        if SCENE_RE.match(ln):
            flush_line()
            elements.append(Element("scene", ln))
            continue
        cue = CUE_RE.match(ln)
        if cue and _is_caps(cue.group(1)) and not FX_PREFIX_RE.match(ln):
            flush_line()
            current_char = cue.group(1).strip()
            rest = cue.group(2).strip()
            if rest:
                buffer.append(rest)
            continue
        if _is_caps(ln):
            flush_line()
            elements.append(Element("direction", FX_PREFIX_RE.sub("", ln).strip()))
            continue
        # dialogue continuation (mixed case)
        if current_char is None:
            elements.append(Element("direction", ln))
        else:
            buffer.append(ln)
    flush_line()

    for i, el in enumerate(elements):
        el.index = i
    return elements


def characters(elements: List[Element]) -> List[str]:
    seen = []
    for el in elements:
        if el.kind == "line" and el.character and el.character not in seen:
            seen.append(el.character)
    return seen


def proper_nouns(elements: List[Element]) -> List[str]:
    """Names Whisper is likely to mangle: character names plus capitalised
    words that appear mid-sentence in dialogue (so not just sentence starts)."""
    names = {c.title() for c in characters(elements)}
    for el in elements:
        if el.kind != "line":
            continue
        # a capitalised word preceded by a lowercase word is a proper noun
        for tok in re.findall(r"\b[a-z]+[,;]?\s+([A-Z][a-z]{2,})\b", el.text):
            names.add(tok)
    return sorted(names)
