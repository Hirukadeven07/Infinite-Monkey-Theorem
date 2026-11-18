"""A deterministic fake policy + PRM, for tests and for trying the UI without a GPU.

Accuracy falls with MATH difficulty level, wrong answers cluster on a few values (so votes
matter), and the fake PRM scores correct final steps higher than wrong ones (so the
verifier methods have something to work with). Nothing here reflects a real model.
"""
from __future__ import annotations

import asyncio
import hashlib
import random
import re
from typing import AsyncIterator

from app.aggregate import answer_key
from app.grading import extract_boxed
from app.prompts import split_steps
from app.schema import Delta, Done, Question, Sampling

P_CORRECT = {1: 0.8, 2: 0.65, 3: 0.5, 4: 0.35, 5: 0.2}
_ASSISTANT_MARK = "</think>\n\n"
_WORD_RE = re.compile(r"\S+\s*|\s+")


def _rng(*parts) -> random.Random:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


class MockBackend:
    kind = "mock"

    def __init__(self, questions: list[Question], seed: int = 0, delay: float = 0.0, steps: int = 4):
        self.questions = questions
        self.seed = seed
        self.delay = delay
        self.steps = steps
        self.info = {"policy_model": "mock", "policy_dtype": "mock", "prm_model": "mock", "prm_precision": "mock",
                     "gpu": "none", "vllm_version": "mock", "mock": True}

    # -- helpers -------------------------------------------------------------
    def _question(self, text: str) -> Question:
        for q in self.questions:
            if q.problem in text:
                return q
        # Custom questions typed into the UI: pretend they are medium difficulty with answer 42.
        return Question("custom", "", "42", 3, "custom")

    def wrong_answers(self, q: Question) -> list[str]:
        """Three plausible-looking wrong answers that are never equivalent to the gold one."""
        gold = answer_key(q.answer)
        return [w for w in ("-1", "0", "7", "12") if answer_key(w) != gold][:3]

    def _answer(self, q: Question, rng: random.Random) -> str:
        if rng.random() < P_CORRECT.get(q.level, 0.5):
            return q.answer
        return rng.choice(self.wrong_answers(q))

    def _step_text(self, q: Question, k: int, rng_parts: tuple) -> str:
        if k < self.steps:
            return f"Step {k}: mock reasoning about the problem, part {k}."
        return f"So the answer is \\boxed{{{self._answer(q, _rng(*rng_parts))}}}."

    def _completion(self, prompt: str, index: int, stop: list[str] | None) -> tuple[str, bool]:
        q = self._question(prompt)
        prefix = prompt.rsplit(_ASSISTANT_MARK, 1)[-1]
        done = len(split_steps(prefix))
        rng_parts = (self.seed, prompt, index)
        if stop and "\n\n" in stop:
            k = done + 1
            return self._step_text(q, k, rng_parts), k >= self.steps
        texts = [self._step_text(q, k, rng_parts) for k in range(done + 1, self.steps + 1)]
        return "\n\n".join(texts), True

    # -- Backend -------------------------------------------------------------
    async def generate(self, prompt: str, n: int, sampling: Sampling, stop: list[str] | None = None,
                       deltas: bool = True) -> AsyncIterator[Delta | Done]:
        outs = [self._completion(prompt, i, stop) for i in range(n)]
        words = [_WORD_RE.findall(text) for text, _ in outs]
        pos = [0] * n
        pending = set(range(n))
        while pending:
            for i in sorted(pending):
                if pos[i] < len(words[i]):
                    if deltas:
                        yield Delta(i, words[i][pos[i]])
                    pos[i] += 1
                if pos[i] >= len(words[i]):
                    pending.discard(i)
                    text, finished = outs[i]
                    yield Done(i, text, len(text.split()), finished)
            await asyncio.sleep(self.delay)

    async def score(self, problem: str, solutions: list[list[str]]) -> list[list[float]]:
        gold = answer_key(self._question(problem).answer)
        out = []
        for steps in solutions:
            scores = []
            for j, step in enumerate(steps):
                rng = _rng(self.seed, "prm", problem, j, step)
                boxed = extract_boxed(step)
                if boxed is None:
                    v = 0.6 + rng.uniform(-0.1, 0.1)
                elif answer_key(boxed) == gold:
                    v = 0.75 + rng.uniform(-0.15, 0.15)
                else:
                    v = 0.4 + rng.uniform(-0.15, 0.15)
                scores.append(min(1.0, max(0.0, v)))
            out.append(scores)
        await asyncio.sleep(0)
        return out

    async def close(self) -> None:
        return None
