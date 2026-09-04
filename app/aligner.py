"""Align a parsed script to Whisper word timestamps.

The script is the reference: we only need enough matching anchor words per
line to place it in time, so Whisper mis-hearings do not matter much. Lines
with no anchors are interpolated between their neighbours and flagged.
"""
import difflib
import re
from typing import Dict, List, Optional, Sequence, Tuple

WORD_RE = re.compile(r"[a-z0-9']+")


def norm_tokens(text: str) -> List[str]:
    return WORD_RE.findall(text.lower().replace("’", "'"))


def _similar(a: str, b: str) -> bool:
    if a == b:
        return True
    if len(a) < 4 or len(b) < 4:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


def align_tokens(script: Sequence[str], heard: Sequence[str]) -> List[Tuple[int, int]]:
    """Monotonic alignment (LCS-style DP with fuzzy equality) -> [(i_script, j_heard)]."""
    n, m = len(script), len(heard)
    if n == 0 or m == 0:
        return []
    # dp[i][j] = best number of matches using script[:i], heard[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        si = script[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            if _similar(si, heard[j - 1]):
                row[j] = prev[j - 1] + 1
            else:
                row[j] = row[j - 1] if row[j - 1] >= prev[j] else prev[j]
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        if _similar(script[i - 1], heard[j - 1]) and dp[i][j] == dp[i - 1][j - 1] + 1:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def place_lines(lines: List[str], words: List[dict]) -> List[dict]:
    """For each script line: start, end, coverage (0-1), heard text, estimated flag.

    ``words`` are Whisper words: {"start", "end", "word"} in absolute seconds.
    """
    script_tokens, owner = [], []
    for li, line in enumerate(lines):
        for tok in norm_tokens(line):
            script_tokens.append(tok)
            owner.append(li)
    heard_tokens = []
    for w in words:
        toks = norm_tokens(w["word"])
        heard_tokens.append(toks[0] if toks else "")

    pairs = align_tokens(script_tokens, heard_tokens)

    per_line: Dict[int, List[int]] = {}
    for si, hj in pairs:
        per_line.setdefault(owner[si], []).append(hj)

    total_tokens = {}
    for li in owner:
        total_tokens[li] = total_tokens.get(li, 0) + 1

    placed: List[Optional[dict]] = [None] * len(lines)
    for li, hits in per_line.items():
        hits.sort()
        start = words[hits[0]]["start"]
        end = words[hits[-1]]["end"]
        # extend to cover unmatched heard words that sit inside this line's span
        heard = " ".join(words[j]["word"].strip() for j in range(hits[0], hits[-1] + 1))
        placed[li] = {
            "start": round(start, 2), "end": round(end, 2),
            "coverage": round(len(hits) / max(1, total_tokens.get(li, 1)), 2),
            "heard": heard, "estimated": False,
        }

    # interpolate unplaced lines between their neighbours
    for li in range(len(lines)):
        if placed[li] is not None:
            continue
        prev_end = next((placed[k]["end"] for k in range(li - 1, -1, -1) if placed[k]), None)
        next_start = next((placed[k]["start"] for k in range(li + 1, len(lines)) if placed[k]), None)
        if prev_end is None and next_start is None:
            start = end = 0.0
        elif prev_end is None:
            start = end = next_start
        elif next_start is None:
            start = end = prev_end
        else:
            start = end = round((prev_end + next_start) / 2, 2)
        placed[li] = {"start": start, "end": end, "coverage": 0.0, "heard": "", "estimated": True}
    return placed  # type: ignore[return-value]
