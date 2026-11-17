"""Scaling curves, per-level tables and the cross-validated compute-optimal line."""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Literal

from app.aggregate import answer_key
from app.schema import SampleRecord

BASELINE = "pass@1"


@dataclass(frozen=True)
class ResultRow:
    question_id: str
    method: str
    budget: int
    accuracy: float
    tokens: int
    seconds: float


def _aggregate_correct(keys: list[str | None], weights: list[float], correct_keys: set[str],
                       subset: list[int]) -> bool:
    totals: dict[str, float] = {}
    first: dict[str, int] = {}
    for pos, i in enumerate(subset):
        k = keys[i]
        if k is None:
            continue
        totals[k] = totals.get(k, 0.0) + weights[i]
        first.setdefault(k, pos)
    if not totals:
        return False
    winner = max(totals, key=lambda k: (totals[k], -first[k]))
    return winner in correct_keys


def subsample_curve(samples: list[SampleRecord], budgets: list[int],
                    method: Literal["majority", "bon"], repeats: int = 100, seed: int = 0) -> dict[int, float]:
    """Expected accuracy of the aggregate over k samples drawn without replacement from the pool."""
    keys = [answer_key(s.answer) for s in samples]
    weights = [1.0] * len(samples) if method == "majority" else [s.prm or 0.0 for s in samples]
    correct_keys = {k for k, s in zip(keys, samples) if s.correct and k is not None}
    rng = random.Random(seed)
    n = len(samples)
    out: dict[int, float] = {}
    for k in budgets:
        if n == 0:
            out[k] = 0.0
        elif k >= n:
            out[k] = float(_aggregate_correct(keys, weights, correct_keys, list(range(n))))
        else:
            hits = sum(_aggregate_correct(keys, weights, correct_keys, sorted(rng.sample(range(n), k)))
                       for _ in range(repeats))
            out[k] = hits / repeats
    return out


def pass_at_1(samples: list[SampleRecord]) -> float:
    return mean(s.correct for s in samples) if samples else 0.0


def pool_rows(question_id: str, samples: list[SampleRecord], budgets: list[int],
              repeats: int = 100, seed: int = 0) -> list[ResultRow]:
    """Rows for majority and best-of-N at each budget plus the pass@1 baseline. Cost per row
    is estimated from the pool: mean tokens per sample x k; seconds is the mean sample latency
    (samples run in parallel, so wall time barely grows with k)."""
    mean_tokens = mean(s.tokens for s in samples) if samples else 0
    mean_secs = mean(s.seconds for s in samples) if samples else 0.0
    rows = [ResultRow(question_id, BASELINE, 1, pass_at_1(samples), round(mean_tokens), mean_secs)]
    for method in ("majority", "bon"):
        for k, acc in subsample_curve(samples, budgets, method, repeats, seed).items():
            rows.append(ResultRow(question_id, method, k, acc, round(mean_tokens * k), mean_secs))
    return rows


def curves(rows: list[ResultRow]) -> dict[str, dict[int, float]]:
    acc: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[r.method][r.budget].append(r.accuracy)
    return {m: {b: mean(v) for b, v in sorted(by_b.items())} for m, by_b in acc.items()}


def level_table(rows: list[ResultRow], levels: dict[str, int]) -> list[dict]:
    acc: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    for r in rows:
        if r.question_id in levels:
            acc[(levels[r.question_id], r.method, r.budget)].append(r.accuracy)
    return [{"level": lv, "method": m, "budget": b, "accuracy": mean(v), "n": len(v)}
            for (lv, m, b), v in sorted(acc.items())]


def fold_assignment(levels: dict[str, int], seed: int = 0) -> dict[str, int]:
    """Two folds, stratified by level."""
    rng = random.Random(seed)
    folds: dict[str, int] = {}
    for lv in sorted(set(levels.values())):
        qids = sorted(q for q, l in levels.items() if l == lv)
        rng.shuffle(qids)
        for i, q in enumerate(qids):
            folds[q] = i % 2
    return folds


def compute_optimal(rows: list[ResultRow], levels: dict[str, int], seed: int = 0) -> dict[int, float]:
    """Pick the best method per (level, budget) on one fold, score it on the other, both ways."""
    folds = fold_assignment(levels, seed)
    table: dict[int, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r.method != BASELINE and r.question_id in levels:
            table[r.budget][r.method][r.question_id] = r.accuracy
    out: dict[int, float] = {}
    for budget in sorted(table):
        by_method = table[budget]
        methods = sorted(by_method)
        scores: list[float] = []
        for fold in (0, 1):
            train = {q for q, f in folds.items() if f != fold}
            for lv in sorted(set(levels.values())):
                test_q = [q for q, f in folds.items() if f == fold and levels[q] == lv]
                if not test_q:
                    continue
                choice = _best(by_method, methods, {q for q in train if levels[q] == lv}) \
                    or _best(by_method, methods, train)
                if choice is None:
                    continue
                scores += [by_method[choice][q] for q in test_q if q in by_method[choice]]
        if scores:
            out[budget] = mean(scores)
    return out


def _best(by_method: dict[str, dict[str, float]], methods: list[str], qids: set[str]) -> str | None:
    best, best_acc = None, -1.0
    for m in methods:
        vals = [by_method[m][q] for q in qids if q in by_method[m]]
        if vals and mean(vals) > best_acc:
            best, best_acc = m, mean(vals)
    return best
