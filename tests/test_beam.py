import pytest

from app.aggregate import weighted_vote
from app.methods.beam import run_beam
from app.mock import MockBackend
from tests.helpers import QS


async def test_beam_completes_and_emits_steps(ctx, events):
    res = await run_beam(ctx, QS[0], 8, m=4)
    steps = [e for e in events if e["type"] == "beam_step"]
    assert steps and all(sum(nd["kept"] for nd in s["nodes"]) <= 2 for s in steps)
    assert len(steps[0]["nodes"]) == 8 and all(nd["parent"] is None for nd in steps[0]["nodes"])
    later = steps[1]["nodes"]
    kept_ids = {nd["id"] for nd in steps[0]["nodes"] if nd["kept"]}
    assert {nd["parent"] for nd in later} <= kept_ids
    assert res.final_answer is not None and res.samples
    assert res.final_answer == weighted_vote([s.answer for s in res.samples], [s.prm for s in res.samples])
    assert any(e["type"] == "baseline" for e in events)
    assert events[-1]["type"] == "run_end"
    assert res.tokens > 0 and res.detail["nodes"]


async def test_beam_budget_validation(ctx):
    with pytest.raises(ValueError):
        await run_beam(ctx, QS[0], 6, m=4)
    with pytest.raises(ValueError):
        await run_beam(ctx, QS[0], 2, m=4)


async def test_beam_respects_max_steps(ctx, events):
    ctx.backend = MockBackend(QS, steps=50)
    res = await run_beam(ctx, QS[0], 4, m=4, max_steps=3)
    assert len([e for e in events if e["type"] == "beam_step"]) == 3
    assert res.samples  # falls back to the active beams
