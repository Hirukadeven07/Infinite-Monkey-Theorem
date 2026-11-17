"""MATH-500 loading, caching and stratified subsets."""
from __future__ import annotations

import json
import random
from pathlib import Path

import httpx

from app.schema import Question

MATH500_URL = "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl"
DEFAULT_PATH = Path("data/math500.jsonl")


def load_math500(path: Path = DEFAULT_PATH) -> list[Question]:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        resp = httpx.get(MATH500_URL, follow_redirects=True, timeout=60)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    questions = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            questions.append(Question(
                id=row["unique_id"], problem=row["problem"], answer=row["answer"],
                level=int(row["level"]), subject=row["subject"],
            ))
    return questions


def stratified_subset(questions: list[Question], per_level: int, seed: int) -> list[Question]:
    rng = random.Random(seed)
    out: list[Question] = []
    for level in sorted({q.level for q in questions}):
        pool = sorted((q for q in questions if q.level == level), key=lambda q: q.id)
        rng.shuffle(pool)
        out.extend(pool[:per_level])
    return out
