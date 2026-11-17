from app.aggregate import majority, weighted_vote
from app.methods.revisions import run_revisions
from tests.helpers import QS


async def test_revisions_sequential_chain(ctx, events):
    res = await run_revisions(ctx, QS[0], 4)
    assert [s.idx for s in res.samples] == [0, 1, 2, 3]
    order = [e["sample"] for e in events if e["type"] in ("sample_delta", "sample_done")]
    assert order == sorted(order)  # strictly sequential: attempt i finishes before i+1 starts
    assert res.final_answer == weighted_vote([s.answer for s in res.samples], [s.prm for s in res.samples])
    assert res.majority_answer == majority([s.answer for s in res.samples])
    base = [e for e in events if e["type"] == "baseline"]
    assert len(base) == 1 and base[0]["correct"] == res.samples[0].correct
    assert events[-1]["type"] == "run_end"


async def test_revision_prompt_receives_previous(ctx, monkeypatch):
    seen = []
    orig = ctx.backend.generate

    def spy(prompt, *a, **k):
        seen.append(prompt)
        return orig(prompt, *a, **k)

    monkeypatch.setattr(ctx.backend, "generate", spy)
    await run_revisions(ctx, QS[0], 3)
    assert len(seen) == 3
    assert "Previous attempt" not in seen[0] and "Previous attempt 2:" in seen[2]
