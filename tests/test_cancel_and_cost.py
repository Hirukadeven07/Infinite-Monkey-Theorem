"""Stop must also stop PRM work; bench beam must not pay for an unused baseline; the live
winner highlight must group answers the same way the vote does."""
import asyncio
import threading
import time

from fastapi.testclient import TestClient

from app.aggregate import answer_key
from app.backends import ColabBackend
from app.bench import BenchConfig, BenchRunner
from app.methods.beam import run_beam
from app.methods.sampling import run_sampling
from app.mock import MockBackend
from app.store import Store
from colab.gateway import create_app
from tests.fakes import INFO, FakePolicy
from tests.helpers import QS


class SlowPrm:
    """Scores in 50 'batches' of 0.05 s, checking should_stop between batches."""

    def __init__(self):
        self.stopped_early = threading.Event()

    def score(self, problem, solutions, should_stop=None):
        for _ in range(50):
            if should_stop and should_stop():
                self.stopped_early.set()
                return []
            time.sleep(0.05)
        return [[0.5] * len(s) for s in solutions]


def test_gateway_cancel_stops_prm_and_frees_it():
    prm = SlowPrm()
    client = TestClient(create_app(FakePolicy(), prm, "t", INFO))
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"op": "auth", "token": "t"})
        ws.receive_json()
        ws.send_json({"op": "score", "job": "s1", "problem": "p", "solutions": [["a"]]})
        time.sleep(0.15)
        ws.send_json({"op": "cancel", "job": "s1"})
        assert prm.stopped_early.wait(2)
        t0 = time.monotonic()
        ws.send_json({"op": "score", "job": "s2", "problem": "p", "solutions": [["a"]]})
        msg = ws.receive_json()
        while msg.get("job") != "s2":
            msg = ws.receive_json()
        assert msg["type"] == "scores" and time.monotonic() - t0 < 4  # lock was released early


async def test_client_cancelled_score_sends_cancel():
    from tests.test_colab_backend import TOKEN
    prm = SlowPrm()
    import socket
    import uvicorn
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(FakePolicy(), prm, TOKEN, INFO), host="127.0.0.1",
                                           port=port, log_level="warning", timeout_graceful_shutdown=1))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    while not server.started:
        await asyncio.sleep(0.02)
    try:
        be = ColabBackend(f"http://127.0.0.1:{port}", TOKEN)
        await be.connect()
        task = asyncio.create_task(be.score("p", [["a"]]))
        await asyncio.sleep(0.2)
        task.cancel()
        assert await asyncio.to_thread(prm.stopped_early.wait, 3)
        await be.close()
    finally:
        server.should_exit = True
        th.join(5)


class CountingBackend(MockBackend):
    def __init__(self):
        super().__init__(QS)
        self.plain_calls = 0

    async def generate(self, prompt, n, sampling, stop=None, deltas=True):
        if stop is None:
            self.plain_calls += 1
        async for ev in super().generate(prompt, n, sampling, stop, deltas=deltas):
            yield ev


async def test_beam_without_baseline_skips_plain_sample(ctx, events):
    spy = CountingBackend()
    ctx.backend = spy
    res = await run_beam(ctx, QS[0], 4, baseline=False)
    assert spy.plain_calls == 0 and res.samples
    assert not [e for e in events if e["type"] == "baseline"]
    assert res.detail["baseline_correct"] is None


async def test_bench_beam_items_skip_baseline(tmp_path, grader):
    spy = CountingBackend()
    runner = BenchRunner(Store(tmp_path / "t.db"), QS, grader)
    cfg = BenchConfig(per_level=1, methods=["beam"], beam_budgets=[4], repeats=10)
    rid = await runner.start(cfg, spy)
    await runner.wait(rid)
    assert spy.plain_calls == 0


async def test_events_carry_answer_keys_for_highlighting(ctx, events):
    res = await run_sampling(ctx, QS[0], 6, "bon")
    done = [e for e in events if e["type"] == "sample_done"]
    assert all(e["key"] == answer_key(e["answer"]) for e in done)
    assert events[-1]["final_key"] == answer_key(res.final_answer)
