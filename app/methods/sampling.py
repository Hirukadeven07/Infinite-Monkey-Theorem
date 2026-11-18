"""Parallel sampling: N independent samples aggregated by majority vote or PRM-weighted
best-of-N (the paper's "best-of-N weighted")."""
from __future__ import annotations

import asyncio
import time
from contextlib import aclosing
from typing import Literal

from app.aggregate import majority, weighted_vote
from app.methods.context import (RunContext, Tracker, answer_correct, emit_run_end,
                                 emit_run_start, finish_sample, finish_samples)
from app.prompts import solve_prompt
from app.schema import Delta, Done, MethodResult, Question


async def run_sampling(ctx: RunContext, q: Question, n: int,
                       method: Literal["majority", "bon"]) -> MethodResult:
    if n < 1:
        raise ValueError("budget must be at least 1")
    await emit_run_start(ctx, q, method, n)
    tracker = Tracker(ctx, weighted=(method == "bon"))
    lock = asyncio.Lock()
    baseline: dict = {}
    t0 = time.monotonic()

    async def complete(idx: int, text: str, tokens: int) -> None:
        record = await finish_sample(ctx, q, idx, text, tokens, time.monotonic() - t0)
        async with lock:
            await tracker.add(record)
            if idx == 0:
                baseline.update(answer=record.answer, correct=record.correct)
                await ctx.emit({"type": "baseline", **baseline})

    tasks: list[asyncio.Task] = []
    prompt = solve_prompt(q.problem)
    if not ctx.stream:
        # Bench: whole completions only, then one batched PRM call for the pool.
        async with aclosing(ctx.backend.generate(prompt, n, ctx.sampling, deltas=False)) as stream:
            dones = [ev async for ev in stream if isinstance(ev, Done)]
        for record in sorted(await finish_samples(ctx, q, dones, t0), key=lambda r: r.idx):
            await tracker.add(record)
            if record.idx == 0:
                baseline.update(answer=record.answer, correct=record.correct)
    else:
        try:
            async with aclosing(ctx.backend.generate(prompt, n, ctx.sampling)) as stream:
                async for ev in stream:
                    if isinstance(ev, Delta):
                        await ctx.emit({"type": "sample_delta", "sample": ev.index, "text": ev.text})
                    else:
                        tasks.append(asyncio.create_task(complete(ev.index, ev.text, ev.tokens)))
            await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                t.cancel()

    samples = sorted(tracker.samples, key=lambda s: s.idx)
    answers = [s.answer for s in samples]
    maj = majority(answers)
    final = maj if method == "majority" else weighted_vote(answers, [s.prm for s in samples])
    result = MethodResult(method, n, final, answer_correct(final, samples), maj, tracker.tokens,
                          tracker.elapsed(), samples, {"baseline_correct": baseline.get("correct", False)})
    await emit_run_end(ctx, result)
    return result
