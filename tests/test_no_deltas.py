"""Bench never shows tokens, so it must not stream them: no per-token deltas anywhere on the
path, and best-of-N scores its whole pool in one PRM call."""
from app.bench import BenchConfig, BenchRunner
from app.methods.beam import run_beam
from app.methods.sampling import run_sampling
from app.mock import MockBackend
from app.prompts import solve_prompt
from app.schema import Delta, Done, Sampling
from app.store import Store
from tests.helpers import QS, collect


class SpyBackend(MockBackend):
    def __init__(self):
        super().__init__(QS)
        self.deltas_flags = []
        self.score_calls = 0

    async def generate(self, prompt, n, sampling, stop=None, deltas=True):
        self.deltas_flags.append(deltas)
        async for ev in super().generate(prompt, n, sampling, stop, deltas=deltas):
            yield ev

    async def score(self, problem, solutions):
        self.score_calls += 1
        return await super().score(problem, solutions)


async def test_mock_without_deltas_yields_only_dones():
    ev = await collect(MockBackend(QS).generate(solve_prompt(QS[0].problem), 3, Sampling(), deltas=False))
    assert ev and all(isinstance(e, Done) for e in ev) and len(ev) == 3


async def test_sampling_without_stream_batches_prm_and_matches_streamed(ctx, events):
    streamed = await run_sampling(ctx, QS[0], 8, "bon")
    spy = SpyBackend()
    ctx.backend, ctx.stream = spy, False
    events.clear()
    quiet = await run_sampling(ctx, QS[0], 8, "bon")
    assert spy.deltas_flags == [False] and spy.score_calls == 1
    assert not [e for e in events if e["type"] == "sample_delta"]
    assert [s.answer for s in quiet.samples] == [s.answer for s in streamed.samples]
    assert quiet.final_answer == streamed.final_answer and quiet.tokens == streamed.tokens


async def test_beam_never_requests_deltas(ctx):
    spy = SpyBackend()
    ctx.backend = spy
    await run_beam(ctx, QS[0], 8)
    assert spy.deltas_flags and not any(spy.deltas_flags)


async def test_bench_never_requests_deltas(tmp_path, grader):
    spy = SpyBackend()
    runner = BenchRunner(Store(tmp_path / "t.db"), QS, grader)
    cfg = BenchConfig(per_level=1, pool_size=4, beam_budgets=[4], revision_budgets=[2], repeats=10)
    rid = await runner.start(cfg, spy)
    await runner.wait(rid)
    assert spy.deltas_flags and not any(spy.deltas_flags)
