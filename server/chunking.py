"""Split long messages into overlapping windows for sql/005_chunks.sql.

No dependencies, so the server and scripts/backfill_chunks.py share one copy -
two chunkers drifting apart would leave backfilled and live-written messages
cut differently.
"""
from __future__ import annotations

CHUNK_MIN_CHARS = 1500   # messages at or under this are searched whole
CHUNK_SIZE      = 1300
CHUNK_OVERLAP   = 200
_MIN_CUT        = 900    # never cut a window shorter than this to find a boundary
_MIN_TAIL       = 300    # a remainder this short is folded into the last window

# Best boundary first: a paragraph, a line, a sentence, a word.
_BOUNDARIES = ("\n\n", "\n", ". ", " ")


def split_chunks(text: str) -> list[tuple[int, str]]:
    """Return [(start_char, chunk_text), ...], or [] for a short message.

    Windows are ~CHUNK_SIZE chars, overlap by ~CHUNK_OVERLAP, and end on the
    best available boundary, so a sentence is rarely split across two chunks
    without also appearing whole in one of them.
    """
    n = len(text)
    if n <= CHUNK_MIN_CHARS:
        return []
    out: list[tuple[int, str]] = []
    start = 0
    while start < n:
        end = start + CHUNK_SIZE
        if n - end < _MIN_TAIL:
            end = n
        else:
            for b in _BOUNDARIES:
                cut = text.rfind(b, start + _MIN_CUT, end)
                if cut != -1:
                    end = cut + len(b)
                    break
        out.append((start, text[start:end]))
        if end >= n:
            break
        # Next window starts CHUNK_OVERLAP back, moved forward to a line or
        # word start so it does not open mid-word.
        nxt = end - CHUNK_OVERLAP
        for b in ("\n", " "):
            cut = text.find(b, nxt, end - CHUNK_OVERLAP // 2)
            if cut != -1:
                nxt = cut + 1
                break
        start = max(nxt, start + 1)
    return out
