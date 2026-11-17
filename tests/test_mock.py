from app.aggregate import answer_key
from app.backends import Backend
from app.grading import extract_boxed
from app.mock import MockBackend
from app.prompts import solve_prompt
from app.schema import Delta, Done, Sampling
from tests.helpers import QS, collect


def dones(events):
    return [e for e in events if isinstance(e, Done)]


def test_mock_satisfies_protocol():
    mb = MockBackend(QS)
    assert isinstance(mb, Backend)
    assert mb.kind == "mock" and mb.info["mock"] is True


async def test_generate_n_streams_and_one_done_each():
    mb = MockBackend(QS)
    ev = await collect(mb.generate(solve_prompt(QS[0].problem), 4, Sampling()))
    ds = dones(ev)
    assert sorted(d.index for d in ds) == [0, 1, 2, 3] and all(d.finished for d in ds)
    assert all(extract_boxed(d.text) for d in ds)
    for d in ds:  # deltas concatenate to the final text
        assert "".join(e.text for e in ev if isinstance(e, Delta) and e.index == d.index) == d.text


async def test_deterministic():
    a = await collect(MockBackend(QS).generate(solve_prompt(QS[0].problem), 3, Sampling()))
    b = await collect(MockBackend(QS).generate(solve_prompt(QS[0].problem), 3, Sampling()))
    assert dones(a) == dones(b)


async def test_easy_questions_mostly_correct_hard_mostly_wrong():
    mb = MockBackend(QS)
    easy = [q for q in QS if q.level == 1][0]
    hard = [q for q in QS if q.level == 5][0]
    async def acc(q):
        ds = dones(await collect(mb.generate(solve_prompt(q.problem), 64, Sampling())))
        return sum(answer_key(extract_boxed(d.text)) == answer_key(q.answer) for d in ds) / 64
    assert await acc(easy) > 0.6 and await acc(hard) < 0.4


async def test_stop_yields_one_step_until_final():
    mb = MockBackend(QS, steps=3)
    prefix = solve_prompt(QS[0].problem)
    finished, n = False, 0
    while not finished:
        d = dones(await collect(mb.generate(prefix, 1, Sampling(), stop=["\n\n"])))[0]
        assert "\n\n" not in d.text
        prefix += d.text + "\n\n"
        finished, n = d.finished, n + 1
    assert n == 3 and extract_boxed(prefix)


async def test_score_shape_and_range():
    s = await MockBackend(QS).score(QS[0].problem, [["a", "b"], ["c"]])
    assert [len(x) for x in s] == [2, 1] and all(0 <= v <= 1 for x in s for v in x)


async def test_score_prefers_correct_final_step():
    mb = MockBackend(QS)
    q = QS[0]
    good = [f"So the answer is \\boxed{{{q.answer}}}."]
    bad = ["So the answer is \\boxed{12345}."]
    sg, sb = await mb.score(q.problem, [good, bad])
    assert sg[-1] > sb[-1]
