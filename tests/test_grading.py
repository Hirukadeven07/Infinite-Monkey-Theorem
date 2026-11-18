import pytest

from app.grading import Grader, extract_boxed, is_correct


def test_extract_last_boxed_nested():
    assert extract_boxed(r"x \boxed{1} then \boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_extract_fbox():
    assert extract_boxed(r"so \fbox{7}.") == "7"


def test_extract_none():
    assert extract_boxed("no answer here") is None


def test_extract_unbalanced_is_none():
    assert extract_boxed(r"\boxed{\frac{1}{2}") is None


@pytest.mark.parametrize("pred,gold", [
    (r"\frac12", r"\frac{1}{2}"),
    ("0.5", r"\frac{1}{2}"),
    (r"\dfrac{3}{4}", r"\frac{3}{4}"),
    ("(1,2)", r"(1, 2)"),
    (r"2\sqrt{2}", r"\sqrt{8}"),
    (r"\left( 3, \frac{\pi}{2} \right)", r"\left( 3, \frac{\pi}{2} \right)"),
])
def test_equivalent(pred, gold):
    assert is_correct(pred, gold)


def test_not_equivalent():
    assert not is_correct("3", "4")


def test_none_and_garbage_are_wrong():
    assert not is_correct(None, "4")
    assert not is_correct("4", r"\frac{{{")  # unparseable gold must not raise


async def test_grader_pool():
    g = Grader(workers=1)
    try:
        assert await g.grade("0.5", r"\frac{1}{2}") is True
        assert await g.grade(None, "1") is False
        assert await g.grade("3", "4") is False
    finally:
        g.close()


async def test_grader_recovers_after_a_hanging_check():
    g = Grader(workers=1, timeout=6.0)  # long enough for a fresh worker to spawn on Windows
    try:
        assert await g.grade(r"10^{10^{10}}", "5") is False   # pathological: hangs for a long time
        assert g.timeouts == 1
        assert await g.grade("0.5", r"\frac{1}{2}") is True  # pool must have been replaced
    finally:
        g.close()


def test_empty_boxed_is_no_answer():
    assert extract_boxed(r"so \boxed{ }") is None
