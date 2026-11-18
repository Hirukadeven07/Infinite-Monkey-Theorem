"""Model backends. Every backend streams samples and scores solutions with a PRM."""
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import suppress
from typing import AsyncIterator, Protocol, runtime_checkable

from app.schema import BackendError, Delta, Done, JobError, Sampling


@runtime_checkable
class Backend(Protocol):
    kind: str   # "colab" | "mock"
    info: dict  # policy_model, prm_model, prm_precision, gpu, vllm_version, mock

    def generate(self, prompt: str, n: int, sampling: Sampling, stop: list[str] | None = None,
                 deltas: bool = True) -> AsyncIterator[Delta | Done]:
        """Interleaved Deltas and exactly one Done per index 0..n-1 (only Dones when
        deltas=False). Closing the iterator early cancels the job."""
        ...

    async def score(self, problem: str, solutions: list[list[str]]) -> list[list[float]]:
        """Per-step PRM scores in [0, 1] for each solution (a list of steps)."""
        ...

    async def close(self) -> None:
        ...


# -- Colab gateway client -------------------------------------------------------

def ws_url(pasted: str) -> str:
    """Normalize a pasted tunnel URL (https://x.trycloudflare.com[/ws][/]) to wss://x/ws.
    Plain ws:// is kept only for local hosts, so the token never crosses the internet unencrypted."""
    s = pasted.strip().rstrip("/")
    if s.endswith("/ws"):
        s = s[:-3].rstrip("/")
    scheme, sep, rest = s.partition("://")
    if not sep:
        scheme, rest = "https", s
    host = rest.split("/")[0].rsplit(":", 1)[0].strip("[]").lower()
    local = host in ("localhost", "127.0.0.1", "::1")
    secure = scheme in ("https", "wss") or not local
    return f"{'wss' if secure else 'ws'}://{rest}/ws"


class ColabBackend:
    """Talks to colab/gateway.py over one WebSocket; every request is a job with its own queue."""

    kind = "colab"

    def __init__(self, url: str, token: str, timeout: float = 20.0):
        self.url = ws_url(url)
        self.token = token
        self.timeout = timeout
        self.info: dict = {}
        self._ws = None
        self._reader: asyncio.Task | None = None
        self._queues: dict[str, asyncio.Queue] = {}
        self._send_lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self._ws is not None

    async def connect(self) -> dict:
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosed

        try:
            ws = await connect(self.url, max_size=None, open_timeout=self.timeout, ping_interval=20,
                               ping_timeout=60)
        except Exception as e:
            raise BackendError(f"could not connect to {self.url}: {e}") from e
        try:
            await ws.send(json.dumps({"op": "auth", "token": self.token}))
            msg = json.loads(await asyncio.wait_for(ws.recv(), self.timeout))
        except ConnectionClosed as e:
            code = e.rcvd.code if e.rcvd else None
            raise BackendError("bad token" if code == 4401 else f"connection closed ({code})") from e
        except Exception as e:
            await ws.close()
            raise BackendError(f"handshake failed: {e}") from e
        if msg.get("type") != "ready":
            await ws.close()
            raise BackendError(f"unexpected handshake reply: {msg}")
        self.info = msg["info"]
        self._ws = ws
        self._reader = asyncio.create_task(self._read(ws))
        return self.info

    async def _read(self, ws) -> None:
        try:
            async for raw in ws:
                msg = json.loads(raw)
                q = self._queues.get(str(msg.get("job")))
                if q is not None:
                    q.put_nowait(msg)
        except Exception:
            pass
        finally:
            if self._ws is ws:
                self._ws = None
            err = BackendError("connection to the Colab gateway was lost")
            for q in self._queues.values():
                q.put_nowait(err)

    async def _send(self, msg: dict) -> None:
        ws = self._ws
        if ws is None:
            raise BackendError("not connected to the Colab gateway")
        try:
            async with self._send_lock:
                await ws.send(json.dumps(msg))
        except Exception as e:
            raise BackendError(f"send failed: {e}") from e

    def _open(self) -> tuple[str, asyncio.Queue]:
        job = uuid.uuid4().hex
        q: asyncio.Queue = asyncio.Queue()
        self._queues[job] = q
        return job, q

    @staticmethod
    def _check(msg) -> dict:
        if isinstance(msg, BackendError):
            raise msg
        if msg.get("type") == "error":
            raise JobError(msg.get("message", "gateway error"))
        return msg

    async def generate(self, prompt: str, n: int, sampling: Sampling, stop: list[str] | None = None,
                       deltas: bool = True) -> AsyncIterator[Delta | Done]:
        job, q = self._open()
        ended = False
        try:
            await self._send({"op": "generate", "job": job, "prompt": prompt, "n": n, "stop": stop, "deltas": deltas,
                              "max_tokens": sampling.max_tokens, "temperature": sampling.temperature,
                              "top_p": sampling.top_p, "top_k": sampling.top_k})
            while True:
                msg = self._check(await q.get())
                kind = msg["type"]
                if kind == "delta":
                    yield Delta(msg["index"], msg["text"])
                elif kind == "done":
                    yield Done(msg["index"], msg["text"], msg["tokens"], msg["finished"])
                elif kind == "end":
                    ended = True
                    return
        finally:
            self._queues.pop(job, None)
            if not ended and self._ws is not None:
                with suppress(Exception):
                    await self._send({"op": "cancel", "job": job})

    async def score(self, problem: str, solutions: list[list[str]]) -> list[list[float]]:
        job, q = self._open()
        answered = False
        try:
            await self._send({"op": "score", "job": job, "problem": problem, "solutions": solutions})
            msg = await q.get()
            answered = True
            return self._check(msg)["scores"]
        finally:
            self._queues.pop(job, None)
            if not answered and self._ws is not None:  # cancelled (e.g. Stop): free the PRM
                with suppress(Exception):
                    await asyncio.shield(self._send({"op": "cancel", "job": job}))

    async def close(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            with suppress(Exception):
                await ws.close()
        if self._reader is not None:
            self._reader.cancel()
