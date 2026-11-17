"""PRM-guided beam search over reasoning steps (the paper's beam search).

Step 0 samples N first steps from the root. Each later step keeps the N/M best incomplete
beams by last-step PRM score and expands each with M continuations, so every step spends
N generations. Beams that reach a final answer (EOS or a \\boxed{}) are set aside; the
final answer is a PRM-weighted vote over them.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import aclosing
from dataclasses import asdict, dataclass

from app.aggregate import majority, weighted_vote
from app.grading import extract_boxed
from app.methods.context import (RunContext, Tracker, answer_correct, emit_run_end,
                                 emit_run_start, finish_sample)
from app.prompts import solve_prompt, split_steps
from app.schema import Done, MethodResult, Question, SampleRecord

STOP = ["\n\n"]


@dataclass
class Node:
    id: str
    parent: str | None
    text: str
    score: float = 0.0
    kept: bool = False
    complete: bool = False
    tokens: int = 0  # tokens along the path from the root


async def _expand(ctx: RunContext, prompt: str, n: int) -> list[Done]:
    out: list[Done] = []
    async with aclosing(ctx.backend.generate(prompt, n, ctx.sampling, stop=STOP)) as stream:
        async for ev in stream:
            if isinstance(ev, Done):
                out.append(ev)
    return sorted(out, key=lambda d: d.index)


async def _baseline(ctx: RunContext, q: Question, prefix: str) -> SampleRecord:
    t0 = time.monotonic()
    dones = await _expand_plain(ctx, prefix)
    d = dones[0]
    rec = await finish_sample(ctx, q, 0, d.text, d.tokens, time.monotonic() - t0)
    await ctx.emit({"type": "baseline", "answer": rec.answer, "correct": rec.correct})
    return rec


async def _expand_plain(ctx: RunContext, prompt: str) -> list[Done]:
    out: list[Done] = []
    async with aclosing(ctx.backend.generate(prompt, 1, ctx.sampling)) as stream:
        async for ev in stream:
            if isinstance(ev, Done):
                out.append(ev)
    return out


async def run_beam(ctx: RunContext, q: Question, n: int, m: int = 4, max_steps: int = 40) -> MethodResult:
    if m < 1 or n < m or n % m:
        raise ValueError(f"beam budget must be a multiple of the beam width {m} and at least {m}")
    await emit_run_start(ctx, q, "beam", n)
    prefix = solve_prompt(q.problem)
    tracker = Tracker(ctx, weighted=True)
    t0 = time.monotonic()
    keep = n // m
    all_nodes: list[Node] = []
    completed: list[Node] = []
    active: list[Node] = [Node("root", None, "")]
    baseline_task = asyncio.create_task(_baseline(ctx, q, prefix))
    try:
        for t in range(max_steps):
            if not active:
                break
            if t == 0:
                groups = [(active[0], await _expand(ctx, prefix, n))]
            else:
                results = await asyncio.gather(*(_expand(ctx, prefix + b.text, m) for b in active))
                groups = list(zip(active, results))
            children: list[Node] = []
            for parent, dones in groups:
                for d in dones:
                    child = Node(
                        id=f"s{t}-{len(children)}",
                        parent=None if parent.id == "root" else parent.id,
                        text=parent.text + d.text + ("" if d.finished else "\n\n"),
                        complete=d.finished or extract_boxed(d.text) is not None,
                        tokens=parent.tokens + d.tokens,
                    )
                    children.append(child)
                    tracker.tokens += d.tokens
            step_lists = [split_steps(c.text) for c in children]
            to_score = [s for s in step_lists if s]
            scored = iter(await ctx.backend.score(q.problem, to_score)) if to_score else iter(())
            prm_steps: list[list[float] | None] = []
            for c, steps in zip(children, step_lists):
                ps = next(scored) if steps else None
                prm_steps.append(ps)
                c.score = ps[-1] if ps else 0.0
            incomplete = sorted((c for c in children if not c.complete), key=lambda c: -c.score)
            for c in incomplete[:keep]:
                c.kept = True
            active = incomplete[:keep]
            all_nodes += children
            await ctx.emit({"type": "beam_step", "step": t,
                            "nodes": [{k: v for k, v in asdict(c).items() if k != "tokens"} for c in children]})
            for c, ps in zip(children, prm_steps):
                if c.complete:
                    completed.append(c)
                    await _record(ctx, q, tracker, c, ps, len(completed) - 1, t0)
            await ctx.emit({"type": "stats", "tokens": tracker.tokens, "seconds": tracker.elapsed()})
            if len(completed) >= n:
                break
        if not completed:
            for i, c in enumerate(active):
                steps = split_steps(c.text)
                ps = (await ctx.backend.score(q.problem, [steps]))[0] if steps else None
                await _record(ctx, q, tracker, c, ps, i, t0)
        baseline = await baseline_task
    finally:
        baseline_task.cancel()

    samples = sorted(tracker.samples, key=lambda s: s.idx)
    answers = [s.answer for s in samples]
    final = weighted_vote(answers, [s.prm for s in samples])
    result = MethodResult("beam", n, final, answer_correct(final, samples), majority(answers),
                          tracker.tokens, tracker.elapsed(), samples,
                          {"nodes": [asdict(c) for c in all_nodes], "beam_width": m,
                           "baseline_correct": baseline.correct})
    await emit_run_end(ctx, result)
    return result


async def _record(ctx: RunContext, q: Question, tracker: Tracker, node: Node,
                  prm_steps: list[float] | None, idx: int, t0: float) -> None:
    answer = extract_boxed(node.text)
    correct = await ctx.grader.grade(answer, q.answer)
    rec = SampleRecord(idx, node.text, answer, correct, prm_steps, node.tokens,
                       round(time.monotonic() - t0, 3))
    await tracker.add(rec, count_tokens=False)
