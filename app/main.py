"""Local web app: serves the UI, streams live runs over a WebSocket, and runs benchmarks.

Run: .venv/Scripts/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
Env: TTS_DATA (MATH-500 jsonl path), TTS_DB (SQLite path), TTS_MOCK_DELAY (seconds per word).
"""
from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import os
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from statistics import mean
from typing import Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.backends import Backend, ColabBackend
from app.bench import BenchConfig, BenchRunner, ConfigMismatch, lock_for
from app.data import load_math500
from app.grading import Grader
from app.methods.beam import run_beam
from app.methods.context import RunContext
from app.methods.revisions import run_revisions
from app.methods.sampling import run_sampling
from app.metrics import BASELINE, compute_optimal, curves, level_table
from app.mock import MockBackend
from app.schema import BackendError, Question, Sampling
from app.store import Store

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
METHODS = ("majority", "bon", "beam", "revisions")


class State:
    questions: list[Question]
    by_id: dict[str, Question]
    backend: Backend
    store: Store
    grader: Grader
    bench: BenchRunner
    sampling: Sampling


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.questions = load_math500(Path(os.environ.get("TTS_DATA", "data/math500.jsonl")))
    state.by_id = {q.id: q for q in state.questions}
    state.store = Store(Path(os.environ.get("TTS_DB", "runs/tts.db")))
    state.grader = Grader()
    state.sampling = Sampling()
    state.backend = MockBackend(state.questions, delay=float(os.environ.get("TTS_MOCK_DELAY", "0.02")))
    state.bench = BenchRunner(state.store, state.questions, state.grader, state.sampling)
    for run in state.store.list_runs("bench"):  # runs interrupted by a restart can be resumed
        if run["status"] == "running":
            state.store.set_status(run["id"], "stopped")
    yield
    await state.backend.close()
    state.grader.close()


app = FastAPI(title="Test-Time Scaling Visualizer", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB_DIR, check_dir=False), name="static")


@app.middleware("http")
async def revalidate_ui(request, call_next):
    """Make browsers revalidate the UI files so an update is never hidden behind a stale cache."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(WEB_DIR / "index.html")


# -- backend -------------------------------------------------------------------

class BackendRequest(BaseModel):
    kind: Literal["colab", "mock"]
    url: str | None = None
    token: str | None = None


def _backend_view() -> dict:
    be = state.backend
    connected = getattr(be, "connected", True)
    return {"kind": be.kind, "connected": connected, "info": be.info}


@app.get("/api/backend")
async def get_backend():
    return _backend_view()


@app.post("/api/backend")
async def set_backend(req: BackendRequest):
    if req.kind == "mock":
        new: Backend = MockBackend(state.questions, delay=float(os.environ.get("TTS_MOCK_DELAY", "0.02")))
    else:
        if not req.url or not req.token:
            raise HTTPException(422, "url and token are required")
        new = ColabBackend(req.url, req.token)
        try:
            await new.connect()
        except BackendError as e:
            raise HTTPException(502, str(e))
    old, state.backend = state.backend, new
    await old.close()
    return _backend_view()


# -- questions -----------------------------------------------------------------

@app.get("/api/questions")
async def questions(level: int | None = None):
    return [q.to_dict() for q in state.questions if level is None or q.level == level]


# -- live ----------------------------------------------------------------------

def _resolve_question(msg: dict) -> Question:
    if msg.get("custom"):
        problem = str(msg["custom"].get("problem", "")).strip()
        answer = str(msg["custom"].get("answer", "")).strip()
        if not problem or not answer:
            raise ValueError("custom questions need a problem and an expected answer")
        qid = "custom-" + hashlib.sha1(problem.encode()).hexdigest()[:10]
        return Question(qid, problem, answer, 0, "custom")
    q = state.by_id.get(str(msg.get("question_id")))
    if q is None:
        raise ValueError("unknown question")
    return q


async def _live_run(msg: dict, emit) -> None:
    try:
        q = _resolve_question(msg)
        method = msg.get("method")
        budget = int(msg.get("budget", 0))
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}")
        backend = state.backend
        ctx = RunContext(backend=backend, grader=state.grader, sampling=state.sampling, emit=emit)
        if method in ("majority", "bon"):
            result = await run_sampling(ctx, q, budget, method)
        elif method == "beam":
            result = await run_beam(ctx, q, budget)
        else:
            result = await run_revisions(ctx, q, budget)
        run_id = state.store.create_run(
            "live", {"question_id": q.id, "method": method, "budget": budget, "mock": bool(backend.info.get("mock")),
                     "final_answer": result.final_answer, "correct": result.correct},
            lock_for(backend.info, state.sampling))
        state.store.set_status(run_id, "done")
        state.store.add_samples(run_id, q.id, method, budget, result.samples)
    except (BackendError, ValueError) as e:
        await emit({"type": "error", "message": str(e)})
    except Exception as e:  # never leave the UI waiting on a run that died
        await emit({"type": "error", "message": f"{type(e).__name__}: {e}"})


@app.websocket("/ws/live")
async def live(ws: WebSocket):
    await ws.accept()
    lock = asyncio.Lock()
    task: asyncio.Task | None = None

    async def emit(event: dict) -> None:
        async with lock:
            await ws.send_json(event)

    async def cancel_current() -> bool:
        nonlocal task
        if task and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
            task = None
            return True
        return False

    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("op") == "run":
                await cancel_current()
                task = asyncio.create_task(_live_run(msg, emit))
            elif msg.get("op") == "stop":
                await cancel_current()
                await emit({"type": "stopped"})
    except WebSocketDisconnect:
        pass
    finally:
        if task and not task.done():
            task.cancel()


# -- bench ---------------------------------------------------------------------

def _require_run(run_id: int) -> dict:
    run = state.store.get_run(run_id)
    if run is None or run["kind"] != "bench":
        raise HTTPException(404, "no such bench run")
    return run


def _require_connected() -> None:
    if not getattr(state.backend, "connected", True):
        raise HTTPException(409, "backend is not connected")


@app.post("/api/bench")
async def start_bench(cfg: BenchConfig):
    _require_connected()
    return {"run_id": await state.bench.start(cfg, state.backend)}


@app.post("/api/bench/{run_id}/resume")
async def resume_bench(run_id: int):
    _require_run(run_id)
    _require_connected()
    try:
        await state.bench.resume(run_id, state.backend)
    except ConfigMismatch as e:
        return JSONResponse(status_code=409, content={"detail": str(e), "keys": e.keys})
    return state.bench.progress(run_id)


@app.post("/api/bench/{run_id}/stop")
async def stop_bench(run_id: int):
    _require_run(run_id)
    state.bench.stop(run_id)
    await state.bench.wait(run_id)
    return state.bench.progress(run_id)


@app.get("/api/runs")
async def runs(kind: Literal["bench", "live"] = "bench"):
    out = state.store.list_runs(kind)
    if kind == "bench":
        for r in out:
            r["progress"] = state.bench.progress(r["id"])
    return out


def _levels() -> dict[str, int]:
    return {q.id: q.level for q in state.questions}


@app.get("/api/bench/{run_id}/metrics")
async def bench_metrics(run_id: int):
    run = _require_run(run_id)
    rows = state.store.rows(run_id)
    levels = _levels()
    cur = curves(rows)
    cost: dict[str, dict[int, dict]] = {}
    by: dict[tuple[str, int], list] = {}
    for r in rows:
        by.setdefault((r.method, r.budget), []).append(r)
    for (method, budget), rs in sorted(by.items()):
        cost.setdefault(method, {})[budget] = {"tokens": round(mean(r.tokens for r in rs)),
                                               "seconds": round(mean(r.seconds for r in rs), 2)}
    return {"progress": state.bench.progress(run_id), "curves": cur,
            "baseline": cur.get(BASELINE, {}).get(1), "by_level": level_table(rows, levels),
            "compute_optimal": compute_optimal(rows, levels, seed=run["config"].get("seed", 0)),
            "cost": cost, "mock": bool(run["config"].get("mock")), "lock": run["lock"], "config": run["config"]}


@app.get("/api/bench/{run_id}/export.csv")
async def export_csv(run_id: int):
    _require_run(run_id)
    levels = _levels()
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["run_id", "question_id", "level", "method", "budget", "accuracy", "tokens", "seconds"])
    for r in state.store.rows(run_id):
        w.writerow([run_id, r.question_id, levels.get(r.question_id, ""), r.method, r.budget,
                    r.accuracy, r.tokens, r.seconds])
    return PlainTextResponse(buf.getvalue(), media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="bench-{run_id}.csv"'})


@app.get("/api/bench/{run_id}/export.json")
async def export_json(run_id: int):
    run = _require_run(run_id)
    levels = _levels()
    rows = [{"question_id": r.question_id, "level": levels.get(r.question_id), "method": r.method,
             "budget": r.budget, "accuracy": r.accuracy, "tokens": r.tokens, "seconds": r.seconds}
            for r in state.store.rows(run_id)]
    return JSONResponse({"run_id": run_id, "config": run["config"], "lock": run["lock"], "rows": rows},
                        headers={"Content-Disposition": f'attachment; filename="bench-{run_id}.json"'})
