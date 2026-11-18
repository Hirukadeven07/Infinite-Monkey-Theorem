"""Fake policy / PRM for exercising the Colab gateway without a GPU."""
import asyncio


class FakePolicy:
    def __init__(self, slow: bool = False):
        self.slow = slow
        self.cancelled = False
        self.calls = []

    async def stream(self, prompt, n, stop, max_tokens, temperature, top_p, top_k):
        self.calls.append({"prompt": prompt, "n": n, "stop": stop})
        if prompt == "FAIL":
            raise RuntimeError("boom")
        chunks = 200 if self.slow else 2
        completed = False
        try:
            texts = [""] * n
            for c in range(chunks):
                for i in range(n):
                    piece = f"w{c} "
                    texts[i] += piece
                    yield {"type": "delta", "index": i, "text": piece}
                await asyncio.sleep(0.02 if self.slow else 0)
            for i in range(n):
                yield {"type": "done", "index": i, "text": texts[i], "tokens": chunks,
                       "finished": not stop}
            completed = True
        finally:
            if not completed:
                self.cancelled = True


class FakePrm:
    def score(self, problem, solutions):
        return [[0.5] * len(s) for s in solutions]


INFO = {"policy_model": "fake", "prm_model": "fake-prm", "prm_precision": "bf16",
        "gpu": "FakeGPU", "vllm_version": "0.0", "policy_dtype": "bfloat16", "mock": False}
