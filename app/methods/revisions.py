"""Sequential revisions: a chain of N attempts, each shown the previous (up to 4) attempts
and asked to correct them.

The paper fine-tuned a revision model and picked answers with a trained verifier; here the
revisions are prompted and the final answer is a PRM-weighted vote over the chain (plain
majority is reported too).
"""
from __future__ import annotations

import time
from contextlib import aclosing

from app.aggregate import majority, weighted_vote
from app.methods.context import (RunContext, Tracker, answer_correct, emit_run_end,
                                 emit_run_start, finish_sample)
from app.prompts import revision_prompt, solve_prompt
from app.schema import Delta, Done, MethodResult, Question


async def run_revisions(ctx: RunContext, q: Question, n: int) -> MethodResult:
    if n < 1:
        raise ValueError("budget must be at least 1")
    await emit_run_start(ctx, q, "revisions", n)
    tracker = Tracker(ctx, weighted=True)
    texts: list[str] = []
    baseline_correct = False
    for i in range(n):
        prompt = solve_prompt(q.problem) if i == 0 else revision_prompt(q.problem, texts)
        t0 = time.monotonic()
        done: Done | None = None
        async with aclosing(ctx.backend.generate(prompt, 1, ctx.sampling)) as stream:
            async for ev in stream:
                if isinstance(ev, Delta):
                    await ctx.emit({"type": "sample_delta", "sample": i, "text": ev.text})
                else:
                    done = ev
        if done is None:
            raise RuntimeError("backend finished without a completion")
        texts.append(done.text)
        record = await finish_sample(ctx, q, i, done.text, done.tokens, time.monotonic() - t0)
        await tracker.add(record)
        if i == 0:
            baseline_correct = record.correct
            await ctx.emit({"type": "baseline", "answer": record.answer, "correct": record.correct})

    samples = tracker.samples
    answers = [s.answer for s in samples]
    final = weighted_vote(answers, [s.prm for s in samples])
    result = MethodResult("revisions", n, final, answer_correct(final, samples), majority(answers),
                          tracker.tokens, tracker.elapsed(), samples,
                          {"baseline_correct": baseline_correct})
    await emit_run_end(ctx, result)
    return result
