"""Colab gateway: one authenticated WebSocket that multiplexes generation and PRM scoring.

Runs inside the Colab notebook next to a vLLM OpenAI-compatible server (the policy) and the
Qwen2.5-Math PRM. Cloudflare quick tunnels don't support SSE, so token streams travel over
this WebSocket instead.

Protocol (JSON messages over /ws):
  client -> {"op": "auth", "token"}                          (first message; bad token closes 4401)
  client -> {"op": "generate", "job", "prompt", "n", "stop", "max_tokens", "temperature", "top_p", "top_k",
              "deltas"}                                      (deltas=false: only "done" messages)
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
import inspect
import json
import secrets
import threading
from contextlib import aclosing, suppress
from typing import AsyncIterator, Callable, Protocol

import httpx
from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect

AUTH_TIMEOUT = 15.0


class PolicyClient(Protocol):
    def stream(self, prompt: str, n: int, stop: list[str] | None, max_tokens: int, temperature: float,
               top_p: float, top_k: int) -> AsyncIterator[dict]: ...


class PrmScorer(Protocol):
    def score(self, problem: str, solutions: list[list[str]],
              should_stop: Callable[[], bool] | None = None) -> list[list[float]]: ...


def create_app(policy: PolicyClient, prm: PrmScorer, token: str, info: dict) -> FastAPI:
    # The tunnel URL is public: no auto-generated docs describing the API to strangers.
    app = FastAPI(title="tts-gateway", docs_url=None, redoc_url=None, openapi_url=None)
    prm_lock = threading.Lock()
    prm_can_stop = "should_stop" in inspect.signature(prm.score).parameters

    def _check(value) -> bool:
        return isinstance(value, str) and secrets.compare_digest(value.encode(), token.encode())

    def _parse(raw: str):
        try:
            msg = json.loads(raw)
        except ValueError:
            return None
        return msg if isinstance(msg, dict) else None

    @app.get("/health")
    async def health(authorization: str | None = Header(default=None)):
        if not authorization or not authorization.startswith("Bearer ") or not _check(authorization[7:]):
            raise HTTPException(401, "bad token")
        return info

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        try:
            first = _parse(await asyncio.wait_for(ws.receive_text(), AUTH_TIMEOUT))
        except Exception:
            first = None
        if first is None or first.get("op") != "auth" or not _check(first.get("token")):
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
                want_deltas = msg.get("deltas", True) is not False
                async with aclosing(stream) as it:
                    async for ev in it:
                        if ev.get("type") == "delta" and not want_deltas:
                            continue  # the tunnel only carries what the client will show
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
            stop = threading.Event()  # set on cancel; the PRM checks it between batches

            def work():
                with prm_lock:
                    if prm_can_stop:
                        return prm.score(msg["problem"], msg["solutions"], should_stop=stop.is_set)
                    return prm.score(msg["problem"], msg["solutions"])
            try:
                scores = await asyncio.to_thread(work)
                await send({"job": job, "type": "scores", "scores": scores})
            except asyncio.CancelledError:
                stop.set()
            except Exception as e:
                with suppress(Exception):
                    await send({"job": job, "type": "error", "message": f"{type(e).__name__}: {e}"})
            finally:
                jobs.pop(job, None)

        await send({"type": "ready", "info": info})
        try:
            while True:
                msg = _parse(await ws.receive_text())
                if msg is None:  # report and keep serving; one bad frame shouldn't kill all jobs
                    await send({"type": "error", "message": "malformed message (expected a JSON object)"})
                    continue
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
            if resp.status_code >= 400:
                detail = (await resp.aread()).decode("utf-8", "replace")[:500]
                raise RuntimeError(f"vLLM HTTP {resp.status_code}: {detail}")
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


class _PrmModel:
    """Qwen2 backbone + the PRM's 2-way scoring head: logits = score(last_hidden_state)."""

    def __init__(self, backbone, head):
        self.backbone, self.head = backbone, head
        self.device = next(backbone.parameters()).device

    def __call__(self, input_ids, attention_mask=None):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        weight = self.head[0].weight
        if weight.device != hidden.device:
            self.head.to(hidden.device)
        return (self.head(hidden.to(weight.dtype)),)


def _checkpoint_file(model_name: str, filename: str) -> str:
    import os

    if os.path.isdir(model_name):
        path = os.path.join(model_name, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(model_name, filename)


def load_prm_model(model_name: str, quant: str = "bf16") -> _PrmModel:
    """Load Qwen2.5-Math-PRM with transformers' built-in Qwen2Model plus its scoring head.

    The checkpoint's own remote code was written for transformers 4.x and breaks on 5.x
    ("'Qwen2RMConfig' object has no attribute 'pad_token_id'"), so it is never used: the
    backbone is a standard Qwen2 model (config.json says model_type "qwen2") and the head is
    score.0 (Linear h->h) -> ReLU -> score.2 (Linear h->2), read straight from the weights.
    quant: "bf16" (>= 40 GB GPUs), "4bit" (smaller GPUs), "fp32" (CPU / tests).
    """
    import json

    import torch
    from safetensors import safe_open
    from transformers import Qwen2Config, Qwen2Model

    dtype = {"bf16": torch.bfloat16, "4bit": torch.float16, "fp32": torch.float32}[quant]
    config = Qwen2Config.from_pretrained(model_name)
    kwargs = {"config": config, "dtype": dtype}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    backbone = Qwen2Model.from_pretrained(model_name, **kwargs).eval()

    try:
        weight_map = json.load(open(_checkpoint_file(model_name, "model.safetensors.index.json")))["weight_map"]
        files = sorted({f for k, f in weight_map.items() if k.startswith("score.")})
    except Exception:  # single-file checkpoint
        files = ["model.safetensors"]
    head_state = {}
    for name in files:
        with safe_open(_checkpoint_file(model_name, name), framework="pt") as f:
            for key in f.keys():
                if key.startswith("score."):
                    head_state[key[len("score."):]] = f.get_tensor(key)
    hidden = config.hidden_size
    head = torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                               torch.nn.Linear(hidden, head_state["2.weight"].shape[0]))
    head.load_state_dict(head_state)
    head = head.to(device=backbone.embed_tokens.weight.device, dtype=dtype).eval()
    return _PrmModel(backbone, head)


class QwenPrm:
    """Qwen2.5-Math-PRM step scoring, following the model card. Imports torch lazily."""

    SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."

    def __init__(self, model_name: str, quant: str = "bf16", batch_size: int = 8):
        import torch
        from transformers import AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(model_name)  # standard Qwen2 tokenizer, no remote code
        self.tok.padding_side = "right"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = load_prm_model(model_name, quant)
        self.sep_id = self.tok.encode("<extra_0>")[0]

    @classmethod
    def from_parts(cls, tok, model, batch_size: int = 8) -> "QwenPrm":
        """Build around an existing tokenizer/model (used by tests)."""
        import torch

        self = cls.__new__(cls)
        self.torch, self.tok, self.model, self.batch_size = torch, tok, model, batch_size
        self.sep_id = tok.encode("<extra_0>")[0]
        return self

    def _text(self, problem: str, steps: list[str]) -> str:
        msgs = [{"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    def score(self, problem: str, solutions: list[list[str]],
              should_stop: Callable[[], bool] | None = None) -> list[list[float]]:
        """Score solutions in right-padded batches; each step's score is P(correct) at its
        <extra_0> marker (causal attention means padding on the right changes nothing).
        Stops between batches once should_stop() is true (the job was cancelled)."""
        torch = self.torch
        out: list[list[float]] = []
        for start in range(0, len(solutions), self.batch_size):
            if should_stop and should_stop():
                break
            chunk = solutions[start:start + self.batch_size]
            enc = self.tok([self._text(problem, s) for s in chunk], return_tensors="pt", padding=True)
            ids = enc.input_ids.to(self.model.device)
            attn = enc.attention_mask.to(self.model.device)
            with torch.no_grad():
                logits = self.model(input_ids=ids, attention_mask=attn)[0]
            probs = torch.softmax(logits.float(), dim=-1)[..., 1]
            mask = (ids == self.sep_id) & attn.bool()
            for row in range(len(chunk)):
                out.append([round(float(p), 6) for p in probs[row][mask[row]].cpu()])
        return out
