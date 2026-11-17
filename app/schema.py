"""Shared data types used across the local app."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Question:
    id: str
    problem: str
    answer: str
    level: int
    subject: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_tokens: int = 2048

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Delta:
    """A chunk of streamed text for one sample."""
    index: int
    text: str


@dataclass(frozen=True)
class Done:
    """A finished sample. finished=False means it stopped at a stop string (step boundary);
    True means EOS or the length limit."""
    index: int
    text: str
    tokens: int
    finished: bool


@dataclass
class SampleRecord:
    idx: int
    text: str
    answer: str | None
    correct: bool
    prm_steps: list[float] | None
    tokens: int
    seconds: float

    @property
    def prm(self) -> float | None:
        return self.prm_steps[-1] if self.prm_steps else None


@dataclass
class MethodResult:
    method: str
    budget: int
    final_answer: str | None
    correct: bool
    majority_answer: str | None
    tokens: int
    seconds: float
    samples: list[SampleRecord] = field(default_factory=list)
    detail: dict = field(default_factory=dict)


class BackendError(Exception):
    """The model backend failed or disconnected."""
