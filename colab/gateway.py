"""Colab gateway: one authenticated WebSocket that multiplexes generation and PRM scoring.

Runs inside the Colab notebook next to a vLLM OpenAI-compatible server (the policy) and the
Qwen2.5-Math PRM. Cloudflare quick tunnels don't support SSE, so token streams travel over
this WebSocket instead.

Protocol (JSON messages over /ws):
  client -> {"op": "auth", "token"}                          (first message; bad token closes 4401)
  client -> {"op": "generate", "job", "prompt", "n", "stop", "max_tokens", "temperature", "top_p", "top_k"}
  client -> {"op": "score", "job", "problem", "solutions": [[step, ...], ...]}
  client -> {"op": "cancel", "job"}
  server -> {"type": "ready", "info"}
  server -> {"job", "type": "delta", "index", "text"}
  server -> {"job", "type": "done", "index", "text", "tokens", "finished"}
  server -> {"job", "type": "end"}                            (after all dones, or after cancel)
  server -> {"job", "type": "scores", "scores": [[float, ...], ...]}
  server -> {"job", "type": "error", "message"}

Module-level imports are stdlib + fastapi + httpx only, so this file is testable without a GPU.
"""
from __future__ import annotations

import asyncio
import json
import secrets
import threading
from contextlib import aclosing, suppress
from typing import AsyncIterator, Protocol

import httpx
from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect

AUTH_TIMEOUT = 15.0


class PolicyClient(Protocol):
    def stream(self, prompt: str, n: int, stop: list[str] | None, max_tokens: int, temperature: float,
               top_p: float, top_k: int) -> AsyncIterator[dict]: ...


class PrmScorer(Protocol):
    def score(self, problem: str, solutions: list[list[str]]) -> list[list[float]]: ...


def create_app(policy: PolicyClient, prm: PrmScorer, token: str, info: dict) -> FastAPI:
    app = FastAPI(title="tts-gateway")
    prm_lock = threading.Lock()

    def _check(value: str | None) -> bool:
        return value is not None and secrets.compare_digest(value.encode(), token.encode())

    @app.get("/health")
    async def health(authorization: str | None = Header(default=None)):
        if not authorization or not authorization.startswith("Bearer ") or not _check(authorization[7:]):
            raise HTTPException(401, "bad token")
        return info

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        try:
            first = await asyncio.wait_for(ws.receive_json(), AUTH_TIMEOUT)
        except Exception:
            await ws.close(code=4401)
            return
        if first.get("op") != "auth" or not _check(first.get("token")):
            await ws.close(code=4401)
            return
        send_lock = asyncio.Lock()
        jobs: dict[str, asyncio.Task] = {}

        async def send(msg: dict) -> None:
            async with send_lock:
                await ws.send_text(json.dumps(msg))

        async def run_generate(job: str, msg: dict) -> None:
            try:
                stream = policy.stream(msg["prompt"], int(msg.get("n", 1)), msg.get("stop"),
                                       int(msg.get("max_tokens", 2048)), float(msg.get("temperature", 0.7)),
                                       float(msg.get("top_p", 0.8)), int(msg.get("top_k", 20)))
                async with aclosing(stream) as it:
                    async for ev in it:
                        await send({"job": job, **ev})
                await send({"job": job, "type": "end"})
            except asyncio.CancelledError:
                with suppress(Exception):  # the socket may already be gone
                    await send({"job": job, "type": "end"})
            except Exception as e:  # report, keep the socket alive
                with suppress(Exception):
                    await send({"job": job, "type": "error", "message": f"{type(e).__name__}: {e}"})
            finally:
                jobs.pop(job, None)

        async def run_score(job: str, msg: dict) -> None:
            def work():
                with prm_lock:
                    return prm.score(msg["problem"], msg["solutions"])
            try:
                scores = await asyncio.to_thread(work)
                await send({"job": job, "type": "scores", "scores": scores})
            except asyncio.CancelledError:
                pass
            except Exception as e:
                with suppress(Exception):
                    await send({"job": job, "type": "error", "message": f"{type(e).__name__}: {e}"})
            finally:
                jobs.pop(job, None)

        await send({"type": "ready", "info": info})
        try:
            while True:
                msg = await ws.receive_json()
                op, job = msg.get("op"), str(msg.get("job", ""))
                if op == "generate":
                    jobs[job] = asyncio.create_task(run_generate(job, msg))
                elif op == "score":
                    jobs[job] = asyncio.create_task(run_score(job, msg))
                elif op == "cancel":
                    task = jobs.get(job)
                    if task:
                        task.cancel()
                    else:
                        await send({"job": job, "type": "end"})
        except WebSocketDisconnect:
            pass
        finally:
            for task in list(jobs.values()):
                task.cancel()

    return app


class VllmPolicy:
    """Streams completions from a vLLM OpenAI-compatible server."""

    def __init__(self, base_url: str, model: str, tokenizer=None, transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.tokenizer = tokenizer
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10), transport=transport)

    def _count(self, text: str, chunks: int) -> int:
        if self.tokenizer is None:
            return chunks
        return len(self.tokenizer(text).input_ids)

    async def stream(self, prompt, n, stop, max_tokens, temperature, top_p, top_k):
        body = {"model": self.model, "prompt": prompt, "n": n, "stop": stop, "max_tokens": max_tokens,
                "temperature": temperature, "top_p": top_p, "top_k": top_k, "stream": True}
        texts: dict[int, str] = {}
        chunks: dict[int, int] = {}
        async with self._client.stream("POST", f"{self.base_url}/v1/completions", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                for ch in json.loads(data).get("choices", []):
                    i = ch["index"]
                    piece = ch.get("text") or ""
                    if piece:
                        texts[i] = texts.get(i, "") + piece
                        chunks[i] = chunks.get(i, 0) + 1
                        yield {"type": "delta", "index": i, "text": piece}
                    if ch.get("finish_reason") is not None:
                        text = texts.get(i, "")
                        # finish_reason "stop" covers both EOS and stop strings; vLLM's stop_reason
                        # names the stop string when one was hit (a step boundary, not the end).
                        hit_stop = ch["finish_reason"] == "stop" and isinstance(ch.get("stop_reason"), str)
                        yield {"type": "done", "index": i, "text": text, "tokens": self._count(text, chunks.get(i, 0)),
                               "finished": not hit_stop}


class QwenPrm:
    """Qwen2.5-Math-PRM step scoring, following the model card. Imports torch lazily."""

    SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."

    def __init__(self, model_name: str, quant: str = "bf16"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        kwargs = {"trust_remote_code": True, "device_map": "auto"}
        if quant == "4bit":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                                                               bnb_4bit_compute_dtype=torch.float16)
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        self.model = AutoModel.from_pretrained(model_name, **kwargs).eval()
        self.sep_id = self.tok.encode("<extra_0>")[0]

    def score(self, problem: str, solutions: list[list[str]]) -> list[list[float]]:
        torch = self.torch
        out = []
        for steps in solutions:
            msgs = [{"role": "system", "content": self.SYSTEM},
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"}]
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            ids = self.tok.encode(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                logits = self.model(input_ids=ids)[0]
            mask = ids == self.sep_id
            probs = torch.softmax(logits.float(), dim=-1)[0][mask[0]][:, 1]
            out.append([round(float(p), 6) for p in probs.cpu()])
        return out
