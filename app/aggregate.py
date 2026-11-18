"""Grouping equivalent answers and voting over them."""
from __future__ import annotations

import re

NO_ANSWER = "(no answer)"

_TEXT_RE = re.compile(r"\\(?:text|textbf|mathrm|mbox)\{([^{}]*)\}")
_DROP = (r"\left", r"\right", r"\!", r"\,", r"\;", r"\:", r"\displaystyle", "$")


def answer_key(ans: str | None) -> str | None:
    """Canonical string used to group answers that are written differently."""
    if ans is None:
        return None
    s = ans
    for _ in range(3):
        s = _TEXT_RE.sub(r"\1", s)
    for tok in _DROP:
        s = s.replace(tok, "")
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = s.replace(r"^{\circ}", "").replace(r"^\circ", "")
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    return s or None


def _groups(answers: list[str | None], weights: list[float]) -> list[dict]:
    """Ordered groups by first occurrence: {key, answer, votes, weight, members}."""
    groups: dict[str | None, dict] = {}
    for i, (ans, w) in enumerate(zip(answers, weights)):
        key = answer_key(ans)
        g = groups.get(key)
        if g is None:
            g = groups[key] = {"key": key, "answer": ans, "votes": 0, "weight": 0.0, "members": [], "first": i}
        g["votes"] += 1
        g["weight"] += float(w)
        g["members"].append(i)
    return list(groups.values())


def _winner(answers: list[str | None], weights: list[float]) -> str | None:
    real = [g for g in _groups(answers, weights) if g["key"] is not None]
    if not real:
        return None
    best = max(real, key=lambda g: (g["weight"], -g["first"]))
    return best["answer"]


def majority(answers: list[str | None]) -> str | None:
    return _winner(answers, [1.0] * len(answers))


def weighted_vote(answers: list[str | None], scores: list[float | None]) -> str | None:
    return _winner(answers, [s or 0.0 for s in scores])


def tally(answers: list[str | None], scores: list[float | None] | None, correct: list[bool]) -> list[dict]:
    weights = [1.0] * len(answers) if scores is None else [s or 0.0 for s in scores]
    groups = _groups(answers, weights)
    groups.sort(key=lambda g: (g["key"] is None, -g["weight"], g["first"]))
    return [{
        "answer": NO_ANSWER if g["key"] is None else g["answer"],
        "votes": g["votes"],
        "weight": round(g["weight"], 6),
        "correct": any(correct[i] for i in g["members"]) if g["key"] is not None else False,
    } for g in groups]
