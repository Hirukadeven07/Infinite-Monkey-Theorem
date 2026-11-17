"""Raw Qwen3 (non-thinking) prompts. Built as strings so the local app needs no tokenizer."""
from __future__ import annotations

PROMPT_VERSION = "v1"

INSTRUCTION = (
    "Solve the following math problem step by step. Separate steps with a blank line. "
    "Put the final answer in \\boxed{}."
)

_TEMPLATE = "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

REVISION_WINDOW = 4


def solve_prompt(problem: str) -> str:
    return _TEMPLATE.format(content=f"{INSTRUCTION}\n\n{problem}")


def revision_prompt(problem: str, previous: list[str]) -> str:
    window = previous[-REVISION_WINDOW:]
    first = len(previous) - len(window) + 1
    parts = [INSTRUCTION, problem]
    parts += [f"Previous attempt {first + i}:\n{text}" for i, text in enumerate(window)]
    parts.append("Check the previous attempts for mistakes and write a corrected, complete solution.")
    return _TEMPLATE.format(content="\n\n".join(parts))


def split_steps(text: str) -> list[str]:
    return [s.strip() for s in text.split("\n\n") if s.strip()]
