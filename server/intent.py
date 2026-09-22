"""Query-intent helpers for /search: recency intent, project-name resolution,
near-duplicate collapse and automated-prompt fingerprints.

Pure functions with no dependencies, so they can be tested without the stack
(python server/test_intent.py).
"""
from __future__ import annotations

import re

# --- recency intent ----------------------------------------------------------
# "When did I last work on X" is not a relevance question. The most relevant
# rows for it are the ones that talk ABOUT X most, which in an archive are
# usually other projects mentioning X in passing - and they can be months old.
# The answer is a date, so it has to be answered with a date order.
_RECENCY_RE = re.compile(
    r"\b("
    r"when\s+(did|was|were|have)\s+(i|we|you)\s+(last|most\s+recently)"
    r"|when\s+(i|we)\s+(last|most\s+recently)"
    r"|(the\s+)?last\s+time\s+(i|we|you)"
    r"|when\s+was\s+the\s+last"
    r"|most\s+recent(ly)?"
    r"|latest"
    r")\b",
    re.I,
)


def recency_intent(q: str) -> bool:
    """True when the question asks for the latest / last / most recent time."""
    return bool(_RECENCY_RE.search(q or ""))


# --- project-name resolution -------------------------------------------------
# Folder names that are containers, not projects. A session started in one of
# them says nothing about subject matter, and the words are generic enough to
# appear in ordinary questions. Extend with ECHOES_PROJECT_IGNORE.
DEFAULT_PROJECT_IGNORE = {
    "unknown", "temp", "tmp", "scratch", "scratchpad", "system32", "desktop",
    "documents", "downloads", "home", "users", "src", "code", "dev", "test",
    "tests", "projects", "repos", "workspace", "skills", "sandbox",
}
PROJECT_MIN_CHARS = 5   # joined key length; shorter names match too much
_MAX_WINDOW = 8         # longest project name, in query tokens


def _tokens(s: str) -> list:
    # camelCase and letter/digit boundaries are word boundaries in a folder
    # name ("BloodClarity" == "blood clarity"), then any non-alphanumeric run.
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s or "")
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]


def project_key(name: str) -> str:
    """Case, hyphen, underscore and space-insensitive key for a project name."""
    return "".join(_tokens(name))


def build_project_index(projects, ignore=()) -> dict:
    """{key: project} for the names eligible to be matched from a question.

    A name is skipped when it is hidden (".claude"), shorter than
    PROJECT_MIN_CHARS once joined, has no letters, or is on the ignore list.
    When two names share a key, the first one seen wins, so pass `projects`
    busiest first."""
    skip = {project_key(x) for x in DEFAULT_PROJECT_IGNORE} | {
        project_key(x) for x in ignore if x}
    out = {}
    for p in projects:
        if not p or p.startswith("."):
            continue
        k = project_key(p)
        if len(k) < PROJECT_MIN_CHARS or not re.search(r"[a-z]", k) or k in skip:
            continue
        out.setdefault(k, p)
    return out


def resolve_project(q: str, index: dict):
    """The project a question names, or None.

    Matches WHOLE query tokens only: a contiguous run of the question's tokens,
    joined, must equal a project's key - so "billing service", "billing-
    service" and "Billing_Service" all resolve, while a project called
    "knurl" never matches inside "knurled". The longest match wins, so a
    question naming "billing-service-refunds" resolves to that and not to
    "billing-service"."""
    toks = _tokens(q)
    best, best_len = None, 0
    for i in range(len(toks)):
        acc = ""
        for j in range(i, min(len(toks), i + _MAX_WINDOW)):
            acc += toks[j]
            p = index.get(acc)
            if p and len(acc) > best_len:
                best, best_len = p, len(acc)
    return best


# --- leading-text fingerprints -----------------------------------------------

def lead_fingerprint(content: str, chars: int) -> str:
    """Normalised leading text: digits masked, whitespace collapsed, lowercased.

    Digits are masked so that templated text differing only in a date or a
    counter ("Date: 2026-04-20" vs "Date: 2026-04-21") collapses to one print.
    Mirrored in SQL by app.AUTOMATION_SQL - keep the two in step."""
    s = (content or "")[: chars + 40]
    s = re.sub(r"[0-9]", "#", s)
    s = re.sub(r"\s+", " ", s).strip(" ").lower()
    return s[:chars]


def collapse_near_duplicates(rows, chars: int) -> list:
    """Keep the best-ranked row of each group sharing a leading fingerprint.
    `rows` must already be in rank order."""
    seen, out = set(), []
    for r in rows:
        fp = lead_fingerprint(r["content"], chars)
        if fp in seen:
            continue
        seen.add(fp)
        out.append(r)
    return out


def reorder(rows, boost_project=None, lift=0, demote=None) -> list:
    """Stable rank adjustment on an already-ranked list.

    Each row's position is its rank index, minus `lift` places when its project
    is `boost_project`, plus a push to the back when `demote(row)` is true. A
    boost, not a filter: a strongly-ranked row from another project still
    appears, just below the boosted ones it no longer outranks."""
    back = len(rows) + lift + 1
    keyed = []
    for i, r in enumerate(rows):
        pos = i
        if boost_project and r["project"] == boost_project:
            pos -= lift + 0.5   # lands AHEAD of the row it now ties with
        if demote and demote(r):
            pos += back
        keyed.append((pos, i, r))
    keyed.sort(key=lambda x: (x[0], x[1]))
    return [r for _, _, r in keyed]
