from pathlib import Path

from app.data import load_math500, stratified_subset
from app.prompts import revision_prompt, solve_prompt, split_steps

FIXTURE = Path(__file__).parent / "fixtures" / "math500_sample.jsonl"


def test_solve_prompt_is_qwen3_non_thinking():
    p = solve_prompt("What is 1+1?")
    assert p.startswith("<|im_start|>user\nSolve the following math problem step by step.")
    assert p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "What is 1+1?" in p


def test_revision_prompt_keeps_last_four():
    p = revision_prompt("Q", [f"att{i}" for i in range(6)])
    assert "att0" not in p and "att1" not in p and "att2" in p and "att5" in p


def test_split_steps_drops_empty():
    assert split_steps("a\n\nb\n\n\n\nc\n\n") == ["a", "b", "c"]


def test_load_math500_from_fixture():
    qs = load_math500(FIXTURE)
    assert len(qs) == 10 and {q.level for q in qs} == {1, 2, 3, 4, 5}
    assert all(q.id and q.answer for q in qs)


def test_stratified_subset_deterministic():
    qs = load_math500(FIXTURE)
    a, b = stratified_subset(qs, 1, seed=0), stratified_subset(qs, 1, seed=0)
    assert [q.id for q in a] == [q.id for q in b]
    assert sorted(q.level for q in a) == [1, 2, 3, 4, 5]
