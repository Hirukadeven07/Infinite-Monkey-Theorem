"""Benchmark runs: resumable, config-locked control tests over a stratified MATH-500 subset."""
from __future__ import annotations

import asyncio
import time
from statistics import mean
from typing import Literal

from pydantic import BaseModel, field_validator

from app.backends import Backend
from app.data import stratified_subset
from app.grading import Grader
from app.methods.beam import run_beam
from app.methods.context import RunContext
from app.methods.revisions import run_revisions
from app.methods.sampling import run_sampling
from app.metrics import ResultRow, pool_rows
from app.prompts import PROMPT_VERSION
from app.schema import BackendError, JobError, Question, Sampling
from app.store import Store

BEAM_WIDTH = 4
LOCK_KEYS = ("policy_model", "policy_dtype", "prm_model", "prm_precision", "vllm_version")


class BenchConfig(BaseModel):
    per_level: int = 10
    seed: int = 0
    pool_size: int = 64
    beam_budgets: list[int] = [4, 16]
    revision_budgets: list[int] = [4, 8]
    methods: list[Literal["pool", "beam", "revisions"]] = ["pool", "beam", "revisions"]
    reference: bool = False
    concurrency: int = 4
    repeats: int = 100

    @field_validator("beam_budgets")
    @classmethod
    def _beam_multiple(cls, v: list[int]) -> list[int]:
        bad = [b for b in v if b < BEAM_WIDTH or b % BEAM_WIDTH]
        if bad:
            raise ValueError(f"beam budgets must be multiples of {BEAM_WIDTH}: {bad}")
        return v

    @field_validator("per_level", "pool_size", "concurrency", "repeats")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be at least 1")
        return v


class ConfigMismatch(Exception):
    def __init__(self, keys: list[str]):
        super().__init__(f"backend config differs from this run's: {', '.join(keys)}")
        self.keys = keys


def lock_for(info: dict, sampling: Sampling) -> dict:
    lock = {k: info.get(k) for k in LOCK_KEYS}
    lock["sampling"] = sampling.to_dict()
    lock["prompt_version"] = PROMPT_VERSION
    return lock


def pool_budgets(pool_size: int) -> list[int]:
    out, k = [], 1
    while k <= pool_size:
        out.append(k)
        k *= 2
    if out[-1] != pool_size:
        out.append(pool_size)
    return out


def bench_items(cfg: BenchConfig, questions: list[Question]) -> list[tuple[Question, str, int]]:
    items = []
    for q in stratified_subset(questions, cfg.per_level, cfg.seed):
        if cfg.reference or "pool" in cfg.methods:
            items.append((q, "pool", cfg.pool_size))
        if cfg.reference:
            continue
        if "beam" in cfg.methods:
            items += [(q, "beam", b) for b in cfg.beam_budgets]
        if "revisions" in cfg.methods:
            items += [(q, "revisions", b) for b in cfg.revision_budgets]
    return items


class BenchRunner:
    def __init__(self, store: Store, questions: list[Question], grader: Grader, sampling: Sampling | None = None):
        self.store = store
        self.questions = questions
        self.grader = grader
        self.sampling = sampling or Sampling()
        self._tasks: dict[int, asyncio.Task] = {}
        self._durations: dict[int, list[float]] = {}
        self._errors: dict[int, str] = {}

    async def start(self, cfg: BenchConfig, backend: Backend) -> int:
        config = {**cfg.model_dump(), "mock": bool(backend.info.get("mock")), "gpu": backend.info.get("gpu")}
        run_id = self.store.create_run("bench", config, lock_for(backend.info, self.sampling))
        self._launch(run_id, cfg, backend)
        return run_id

    async def resume(self, run_id: int, backend: Backend) -> None:
        run = self.store.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        if run_id in self._tasks and not self._tasks[run_id].done():
            return
        current = lock_for(backend.info, self.sampling)
        diff = [k for k in run["lock"] if run["lock"][k] != current.get(k)]
        if diff:
            raise ConfigMismatch(diff)
        cfg = BenchConfig(**{k: v for k, v in run["config"].items() if k in BenchConfig.model_fields})
        self._launch(run_id, cfg, backend)

    def stop(self, run_id: int) -> None:
        task = self._tasks.get(run_id)
        if task and not task.done():
            task.cancel()

    async def wait(self, run_id: int) -> None:
        task = self._tasks.get(run_id)
        if task:
            try:
                await task
            except asyncio.CancelledError:
                pass

    def progress(self, run_id: int) -> dict:
        run = self.store.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        cfg = BenchConfig(**{k: v for k, v in run["config"].items() if k in BenchConfig.model_fields})
        total = len(bench_items(cfg, self.questions))
        done = len(self.store.done_items(run_id))
        durs = self._durations.get(run_id) or []
        eta = (mean(durs) * (total - done) / cfg.concurrency) if durs and run["status"] == "running" else None
        return {"status": run["status"], "done": done, "total": total, "failed": self.store.failed_count(run_id),
                "eta_seconds": round(eta) if eta is not None else None, "error": self._errors.get(run_id)}

    # -- internals -----------------------------------------------------------
    def _launch(self, run_id: int, cfg: BenchConfig, backend: Backend) -> None:
        self._errors.pop(run_id, None)
        self.store.set_status(run_id, "running")
        self._tasks[run_id] = asyncio.create_task(self._run(run_id, cfg, backend))

    async def _run(self, run_id: int, cfg: BenchConfig, backend: Backend) -> None:
        done = self.store.done_items(run_id)
        todo = [it for it in bench_items(cfg, self.questions) if (it[0].id, it[1], it[2]) not in done]
        sem = asyncio.Semaphore(cfg.concurrency)
        ctx = RunContext(backend=backend, grader=self.grader, sampling=self.sampling, stream=False)

        async def one(q: Question, item: str, budget: int) -> None:
            async with sem:
                t0 = time.monotonic()
                try:
                    await self._run_item(run_id, cfg, ctx, q, item, budget)
                except JobError as e:
                    # Deterministic per-request failure (e.g. prompt too long): retrying on
                    # resume would fail again, so record it and leave it out of the metrics.
                    self.store.mark_done(run_id, q.id, item, budget, ctx.backend.info.get("gpu"),
                                         {"failed": str(e)})
                self._durations.setdefault(run_id, []).append(time.monotonic() - t0)

        try:
            async with asyncio.TaskGroup() as tg:
                for q, item, budget in todo:
                    tg.create_task(one(q, item, budget))
        except asyncio.CancelledError:
            self.store.set_status(run_id, "stopped")
            raise
        except ExceptionGroup as eg:
            backend_errors = eg.subgroup(BackendError)
            first = (backend_errors or eg).exceptions[0]
            while isinstance(first, ExceptionGroup):
                first = first.exceptions[0]
            # A dropped backend is recoverable (resume later); anything else is a bug.
            self._errors[run_id] = str(first) if backend_errors else repr(first)
            self.store.set_status(run_id, "paused" if backend_errors else "error")
            return
        self.store.set_status(run_id, "done")

    async def _run_item(self, run_id: int, cfg: BenchConfig, ctx: RunContext,
                        q: Question, item: str, budget: int) -> None:
        detail: dict = {}
        if item == "pool":
            res = await run_sampling(ctx, q, budget, "bon")
            rows = pool_rows(q.id, res.samples, pool_budgets(budget), cfg.repeats, cfg.seed)
        elif item == "beam":
            res = await run_beam(ctx, q, budget, m=BEAM_WIDTH)
            rows = [ResultRow(q.id, "beam", budget, float(res.correct), res.tokens, res.seconds)]
            detail = {"nodes": res.detail.get("nodes", [])}
        else:
            res = await run_revisions(ctx, q, budget)
            rows = [ResultRow(q.id, "revisions", budget, float(res.correct), res.tokens, res.seconds)]
        detail["final_answer"] = res.final_answer
        self.store.add_samples(run_id, q.id, item, budget, res.samples)
        self.store.save_rows(run_id, rows)
        self.store.mark_done(run_id, q.id, item, budget, ctx.backend.info.get("gpu"), detail)
