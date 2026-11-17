"""Shared plumbing for the test-time scaling methods: run context, sample finishing,
and live tally/curve tracking. Every method streams events through ctx.emit."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from app.aggregate import NO_ANSWER, answer_key, tally
from app.backends import Backend
from app.grading import Grader, extract_boxed
from app.prompts import split_steps
from app.schema import Question, SampleRecord, Sampling


async def _noop(event: dict) -> None:
    return None


@dataclass
class RunContext:
    backend: Backend
    grader: Grader
    sampling: Sampling = field(default_factory=Sampling)
    emit: Callable[[dict], Awaitable[None]] = _noop


async def finish_sample(ctx: RunContext, q: Question, idx: int, text: str,
                        tokens: int, seconds: float) -> SampleRecord:
    """Extract, grade and PRM-score one completed sample."""
    answer = extract_boxed(text)
    correct = await ctx.grader.grade(answer, q.answer)
    steps = split_steps(text)
    prm_steps = (await ctx.backend.score(q.problem, [steps]))[0] if steps else None
    return SampleRecord(idx, text, answer, correct, prm_steps, tokens, round(seconds, 3))


def answer_correct(answer: str | None, samples: list[SampleRecord]) -> bool:
    """Whether the aggregated answer is correct: some sample in its group was graded correct."""
    key = answer_key(answer)
    return key is not None and any(s.correct for s in samples if answer_key(s.answer) == key)


def winner_correct(entries: list[dict]) -> bool:
    return bool(entries) and entries[0]["answer"] != NO_ANSWER and entries[0]["correct"]


class Tracker:
    """Collects completed samples in completion order and emits the live views."""

    def __init__(self, ctx: RunContext, weighted: bool):
        self.ctx = ctx
        self.weighted = weighted
        self.samples: list[SampleRecord] = []
        self.points: list[dict] = []
        self.tokens = 0
        self.t0 = time.monotonic()

    def elapsed(self) -> float:
        return round(time.monotonic() - self.t0, 3)

    def entries(self) -> list[dict]:
        return tally([s.answer for s in self.samples],
                     [s.prm for s in self.samples] if self.weighted else None,
                     [s.correct for s in self.samples])

    async def add(self, s: SampleRecord, count_tokens: bool = True) -> None:
        """count_tokens=False when the caller accounts generation cost itself (beam search,
        where a completed beam's tokens are shared with its ancestors)."""
        self.samples.append(s)
        if count_tokens:
            self.tokens += s.tokens
        emit = self.ctx.emit
        await emit({"type": "sample_done", "sample": s.idx, "answer": s.answer, "correct": s.correct,
                    "prm": s.prm, "tokens": s.tokens})
        entries = self.entries()
        await emit({"type": "tally", "entries": entries})
        self.points.append({"k": len(self.samples), "correct": winner_correct(entries)})
        await emit({"type": "agg_curve", "points": list(self.points)})
        await emit({"type": "stats", "tokens": self.tokens, "seconds": self.elapsed()})


async def emit_run_start(ctx: RunContext, q: Question, method: str, budget: int) -> None:
    await ctx.emit({"type": "run_start", "method": method, "budget": budget, "question": q.to_dict()})


async def emit_run_end(ctx: RunContext, result) -> None:
    await ctx.emit({"type": "run_end", "final_answer": result.final_answer, "correct": result.correct,
                    "majority_answer": result.majority_answer, "tokens": result.tokens,
                    "seconds": result.seconds})
