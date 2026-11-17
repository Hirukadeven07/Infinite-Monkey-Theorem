from app.aggregate import majority, weighted_vote
from app.methods.sampling import run_sampling
from tests.helpers import QS


async def test_bon_runs_and_emits(ctx, events):
    res = await run_sampling(ctx, QS[0], 8, "bon")
    assert len(res.samples) == 8 and all(s.prm is not None for s in res.samples)
    types = [e["type"] for e in events]
    assert types[0] == "run_start" and types[-1] == "run_end"
    assert types.count("sample_done") == 8 and types.count("tally") == 8
    assert events[-1]["final_answer"] == res.final_answer
    assert res.final_answer == weighted_vote([s.answer for s in res.samples], [s.prm for s in res.samples])


async def test_majority_final_matches_aggregate(ctx):
    res = await run_sampling(ctx, QS[0], 8, "majority")
    assert res.final_answer == majority([s.answer for s in res.samples])
    assert res.majority_answer == res.final_answer


async def test_tokens_counted(ctx):
    res = await run_sampling(ctx, QS[0], 4, "bon")
    assert res.tokens == sum(s.tokens for s in res.samples) > 0


async def test_agg_curve_and_baseline(ctx, events):
    res = await run_sampling(ctx, QS[0], 6, "majority")
    curves = [e for e in events if e["type"] == "agg_curve"]
    assert [p["k"] for p in curves[-1]["points"]] == [1, 2, 3, 4, 5, 6]
    assert curves[-1]["points"][-1]["correct"] == res.correct
    base = [e for e in events if e["type"] == "baseline"]
    assert len(base) == 1 and base[0]["correct"] == res.samples[0].correct
    assert res.detail["baseline_correct"] == res.samples[0].correct


async def test_deltas_streamed_before_done(ctx, events):
    await run_sampling(ctx, QS[0], 2, "bon")
    types = [e["type"] for e in events]
    assert types.index("sample_delta") < types.index("sample_done")
