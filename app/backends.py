"""Model backends. Every backend streams samples and scores solutions with a PRM."""
from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from app.schema import Delta, Done, Sampling


@runtime_checkable
class Backend(Protocol):
    kind: str   # "colab" | "mock"
    info: dict  # policy_model, prm_model, prm_precision, gpu, vllm_version, mock

    def generate(self, prompt: str, n: int, sampling: Sampling,
                 stop: list[str] | None = None) -> AsyncIterator[Delta | Done]:
        """Interleaved Deltas and exactly one Done per index 0..n-1.
        Closing the iterator early cancels the job."""
        ...

    async def score(self, problem: str, solutions: list[list[str]]) -> list[list[float]]:
        """Per-step PRM scores in [0, 1] for each solution (a list of steps)."""
        ...

    async def close(self) -> None:
        ...
