"""Tests for server/intent.py. No dependencies: python server/test_intent.py"""
from intent import (build_project_index, collapse_near_duplicates,
                    lead_fingerprint, project_words, recency_intent, reorder,
                    resolve_project)

PROJECTS = ["billing-service", "billing-service-refunds", "Temp", "scratchpad",
            ".claude", "Jo-Bloggs", "BloodTracker", "api-3.0", "gizmo", "tool"]
IDX = build_project_index(PROJECTS, ignore=["Jo-Bloggs"])


def test_recency_intent():
    yes = ["when I last worked on the billing service",
           "When did we last touch the deploy script?",
           "what's the latest on the migration",
           "most recent session about auth",
           "the last time I ran the backfill"]
    no = ["when did we adopt the cross-encoder",
          "how does the pair gate work",
          "what did we decide about the last mile"]
    assert all(recency_intent(q) for q in yes), [q for q in yes if not recency_intent(q)]
    assert not any(recency_intent(q) for q in no), [q for q in no if recency_intent(q)]


def test_resolve_spellings():
    for q in ["when did I last work on billing service",
              "Billing-Service deploy", "the billing_service repo"]:
        assert resolve_project(q, IDX) == "billing-service", q
    assert resolve_project("blood tracker charts", IDX) == "BloodTracker"
    assert resolve_project("API 3.0 routes", IDX) == "api-3.0"


def test_resolve_longest_wins():
    assert resolve_project("billing service refunds bug", IDX) == "billing-service-refunds"


def test_resolve_rejects_generic_and_partial():
    assert resolve_project("save it in temp", IDX) is None           # ignore list
    assert resolve_project("my scratchpad notes", IDX) is None       # ignore list
    assert resolve_project("the .claude folder", IDX) is None        # hidden
    assert resolve_project("ask jo bloggs", IDX) is None        # caller ignore
    assert resolve_project("which tool", IDX) is None               # under 5 chars
    assert resolve_project("the gizmos edge", IDX) is None          # whole tokens only
    assert resolve_project("gizmo release", IDX) == "gizmo"


def test_project_words():
    assert project_words("BloodTracker-v2") == "blood tracker v2"


def test_fingerprint_masks_digits():
    a = "Write the log.  Project: x\nDate: 2026-04-20 and more"
    b = "Write the log. Project: x Date: 2026-04-21 and more"
    assert lead_fingerprint(a, 200) == lead_fingerprint(b, 200)


def test_collapse_keeps_best_ranked():
    rows = [{"id": 1, "content": "Template A 2026-01-01"},
            {"id": 2, "content": "something else"},
            {"id": 3, "content": "Template A 2026-01-02"}]
    assert [r["id"] for r in collapse_near_duplicates(rows, 200)] == [1, 2]


def test_reorder_boost_and_demote():
    rows = [{"id": i, "project": p} for i, p in
            enumerate(["x", "x", "x", "y", "x", "y"])]
    out = reorder(rows, boost_project="y", lift=2)
    assert [r["id"] for r in out] == [0, 3, 1, 2, 5, 4]   # a boost, not a filter
    out = reorder(rows, demote=lambda r: r["id"] == 0)
    assert [r["id"] for r in out] == [1, 2, 3, 4, 5, 0]


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); n += 1
    print(f"{n} tests passed")
