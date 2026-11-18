"""Answer extraction and equivalence checking with math-verify.

math-verify's default timeouts use Unix signal alarms, which do not exist on Windows and
only work in the main thread, so they are disabled here; Grader enforces a per-task
timeout by running checks in a process pool instead.
"""
from __future__ import annotations

import asyncio
import logging
import re
from concurrent.futures import ProcessPoolExecutor

# Timeouts are deliberately disabled (see module docstring); silence the per-call notice.
logging.getLogger("math_verify").setLevel(logging.ERROR)

_BOX_RE = re.compile(r"\\(?:boxed|fbox)\s*\{")


def extract_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{...} / \\fbox{...} with balanced braces."""
    starts = [m.end() for m in _BOX_RE.finditer(text)]
    for start in reversed(starts):
        depth = 1
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i].strip() or None
    return None


def is_correct(pred: str | None, gold: str) -> bool:
    if pred is None or not pred.strip():
        return False
    try:
        from math_verify import parse, verify

        gold_parsed = parse(f"\\boxed{{{gold}}}", parsing_timeout=None)
        pred_parsed = parse(f"\\boxed{{{pred}}}", parsing_timeout=None)
        if not gold_parsed or not pred_parsed:
            return False
        return bool(verify(gold_parsed, pred_parsed, timeout_seconds=None))
    except Exception:
        return False


class Grader:
    """Async grading in a small process pool, with a cache and a per-task timeout."""

    def __init__(self, workers: int = 2, timeout: float = 10.0):
        self._workers = workers
        self._pool = ProcessPoolExecutor(max_workers=workers)
        self._timeout = timeout
        self._cache: dict[tuple[str | None, str], bool] = {}
        self.timeouts = 0  # checks abandoned as too slow (graded wrong)

    async def grade(self, pred: str | None, gold: str) -> bool:
        if pred is None:
            return False
        key = (pred, gold)
        if key in self._cache:
            return self._cache[key]
        loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(self._pool, is_correct, pred, gold), self._timeout)
        except TimeoutError:
            # The worker keeps computing after wait_for gives up; replace the pool so a few
            # pathological answers can't block grading (and mark everything wrong) for good.
            self.timeouts += 1
            self._replace_pool()
            result = False
        except Exception:
            result = False
        self._cache[key] = result
        return result

    @staticmethod
    def _kill(pool: ProcessPoolExecutor) -> None:
        procs = list((getattr(pool, "_processes", None) or {}).values())
        pool.shutdown(wait=False, cancel_futures=True)
        for p in procs:
            if p.is_alive():
                p.terminate()

    def _replace_pool(self) -> None:
        old, self._pool = self._pool, ProcessPoolExecutor(max_workers=self._workers)
        self._kill(old)

    def close(self) -> None:
        self._kill(self._pool)
