import asyncio

import pytest

from app.bench import BenchConfig, BenchRunner, ConfigMismatch
from app.metrics import curves
from app.mock import MockBackend
from app.schema import BackendError
from app.store import Store
from tests.helpers import QS

CFG = BenchConfig(per_level=1, pool_size=4, beam_budgets=[4], revision_budgets=[2], concurrency=2, repeats=20)
TOTAL = 5 * 3


class FlakyBackend(MockBackend):
    """Mock backend whose generate() raises BackendError after `fail_after` calls."""

    def __init__(self, fail_after: int):
        super().__init__(QS)
        self.calls = 0
        self.fail_after = fail_after

    async def generate(self, prompt, n, sampling, stop=None, deltas=True):
        self.calls += 1
        if self.calls > self.fail_after:
            raise BackendError("tunnel dropped")
        async for ev in super().generate(prompt, n, sampling, stop, deltas=deltas):
            yield ev


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "t.db")


@pytest.fixture
def runner(store, grader):
    return BenchRunner(store, QS, grader)


async def test_bench_completes_and_rows_present(runner, store):
    rid = await runner.start(CFG, MockBackend(QS))
    await runner.wait(rid)
    assert store.get_run(rid)["status"] == "done"
    assert len(store.done_items(rid)) == TOTAL
    c = curves(store.rows(rid))
    assert {"majority", "bon", "pass@1", "beam", "revisions"} <= set(c)
    assert set(c["bon"]) == {1, 2, 4}
    assert runner.progress(rid)["done"] == TOTAL == runner.progress(rid)["total"]
    assert store.get_run(rid)["config"]["mock"] is True


async def test_backend_error_pauses_run(runner, store):  # Review Focus 2
    rid = await runner.start(CFG, FlakyBackend(fail_after=3))
    await runner.wait(rid)
    assert store.get_run(rid)["status"] == "paused"
    assert len(store.done_items(rid)) < TOTAL
    assert "tunnel dropped" in runner.progress(rid)["error"]


async def test_resume_skips_done_items(runner, store, monkeypatch):
    rid = await runner.start(CFG, FlakyBackend(fail_after=3))
    await runner.wait(rid)
    remaining_pool = 5 - len([d for d in store.done_items(rid) if d[1] == "pool"])
    import app.bench as bench
    calls = []
    orig = bench.run_sampling

    async def spy(*a, **k):
        calls.append(a[1].id)
        return await orig(*a, **k)

    monkeypatch.setattr(bench, "run_sampling", spy)
    await runner.resume(rid, MockBackend(QS))
    await runner.wait(rid)
    assert store.get_run(rid)["status"] == "done"
    assert len(calls) == remaining_pool and len(store.done_items(rid)) == TOTAL


async def test_resume_refuses_config_mismatch(runner):  # Review Focus 5
    rid = await runner.start(CFG, MockBackend(QS))
    await runner.wait(rid)
    other = MockBackend(QS)
    other.info = {**other.info, "prm_precision": "4bit"}
    with pytest.raises(ConfigMismatch) as e:
        await runner.resume(rid, other)
    assert e.value.keys == ["prm_precision"]


async def test_reference_run_only_pools(runner, store):
    rid = await runner.start(CFG.model_copy(update={"reference": True}), MockBackend(QS))
    await runner.wait(rid)
    assert {d[1] for d in store.done_items(rid)} == {"pool"}


def test_config_rejects_bad_beam_budget():
    with pytest.raises(ValueError):
        BenchConfig(beam_budgets=[6])


async def test_stop_marks_run_stopped_and_resumable(runner, store):
    rid = await runner.start(CFG, MockBackend(QS, delay=0.01))
    await asyncio.sleep(0.05)
    runner.stop(rid)
    await runner.wait(rid)
    assert store.get_run(rid)["status"] == "stopped"
    await runner.resume(rid, MockBackend(QS))
    await runner.wait(rid)
    assert store.get_run(rid)["status"] == "done" and len(store.done_items(rid)) == TOTAL


async def test_resume_refuses_policy_dtype_change(runner):
    rid = await runner.start(CFG, MockBackend(QS))
    await runner.wait(rid)
    other = MockBackend(QS)
    other.info = {**other.info, "policy_dtype": "half"}
    with pytest.raises(ConfigMismatch) as e:
        await runner.resume(rid, other)
    assert e.value.keys == ["policy_dtype"]


class JobFailBackend(MockBackend):
    """Fails deterministically (like a vLLM context-length 400) for one question only."""

    def __init__(self, bad_problem):
        super().__init__(QS)
        self.bad = bad_problem

    async def generate(self, prompt, n, sampling, stop=None, deltas=True):
        from app.schema import JobError
        if self.bad in prompt:
            raise JobError("context length exceeded")
        async for ev in super().generate(prompt, n, sampling, stop, deltas=deltas):
            yield ev


async def test_job_error_fails_item_but_run_completes(runner, store):
    from app.data import stratified_subset
    bad = stratified_subset(QS, 1, 0)[0]
    rid = await runner.start(CFG, JobFailBackend(bad.problem))
    await runner.wait(rid)
    p = runner.progress(rid)
    assert store.get_run(rid)["status"] == "done"
    assert p["failed"] == 3 and p["done"] == TOTAL
    assert not [r for r in store.rows(rid) if r.question_id == bad.id]
