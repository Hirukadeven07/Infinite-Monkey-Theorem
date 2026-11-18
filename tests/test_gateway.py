import json

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from colab.gateway import VllmPolicy, create_app
from tests.fakes import INFO, FakePolicy, FakePrm

TOKEN = "secret-token"


def make(policy=None):
    policy = policy or FakePolicy()
    return TestClient(create_app(policy, FakePrm(), TOKEN, INFO)), policy


def auth(ws):
    ws.send_json({"op": "auth", "token": TOKEN})
    msg = ws.receive_json()
    assert msg == {"type": "ready", "info": INFO}


def until_end(ws, job):
    out = []
    while True:
        msg = ws.receive_json()
        if msg.get("job") == job:
            out.append(msg)
            if msg["type"] in ("end", "error"):
                return out


def test_bad_token_rejected():
    client, _ = make()
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"op": "auth", "token": "wrong"})
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
        assert e.value.code == 4401


def test_generate_streams_then_end():
    client, _ = make()
    with client.websocket_connect("/ws") as ws:
        auth(ws)
        ws.send_json({"op": "generate", "job": "j1", "prompt": "p", "n": 2, "stop": None, "max_tokens": 10,
                      "temperature": 0.7, "top_p": 0.8, "top_k": 20})
        msgs = until_end(ws, "j1")
        assert {m["index"] for m in msgs if m["type"] == "delta"} == {0, 1}
        dones = [m for m in msgs if m["type"] == "done"]
        assert len(dones) == 2 and dones[0]["text"] == "w0 w1 " and dones[0]["finished"] is True
        assert msgs[-1] == {"job": "j1", "type": "end"}


def test_score_roundtrip():
    client, _ = make()
    with client.websocket_connect("/ws") as ws:
        auth(ws)
        ws.send_json({"op": "score", "job": "s1", "problem": "p", "solutions": [["a", "b"], ["c"]]})
        msg = ws.receive_json()
        assert msg == {"job": "s1", "type": "scores", "scores": [[0.5, 0.5], [0.5]]}


def test_concurrent_jobs_multiplexed():
    client, _ = make(FakePolicy(slow=False))
    with client.websocket_connect("/ws") as ws:
        auth(ws)
        for job in ("a", "b"):
            ws.send_json({"op": "generate", "job": job, "prompt": job, "n": 1, "stop": ["\n\n"],
                          "max_tokens": 10, "temperature": 0.7, "top_p": 0.8, "top_k": 20})
        ends = set()
        while len(ends) < 2:
            msg = ws.receive_json()
            if msg["type"] == "end":
                ends.add(msg["job"])
        assert ends == {"a", "b"}


def test_cancel_sends_end_and_stops():  # Review Focus 3
    client, policy = make(FakePolicy(slow=True))
    with client.websocket_connect("/ws") as ws:
        auth(ws)
        ws.send_json({"op": "generate", "job": "c1", "prompt": "p", "n": 1, "stop": None, "max_tokens": 10,
                      "temperature": 0.7, "top_p": 0.8, "top_k": 20})
        assert ws.receive_json()["type"] == "delta"
        ws.send_json({"op": "cancel", "job": "c1"})
        msgs = until_end(ws, "c1")
        assert msgs[-1]["type"] == "end"
        assert not any(m["type"] == "done" for m in msgs)
    assert policy.cancelled


def test_health_requires_bearer():
    client, _ = make()
    assert client.get("/health").status_code == 401
    assert client.get("/health", headers={"Authorization": "Bearer nope"}).status_code == 401
    r = client.get("/health", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200 and r.json() == INFO


def _sse(chunks):
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.MockTransport(lambda req: httpx.Response(200, text=body,
                                                           headers={"content-type": "text/event-stream"}))


async def test_vllm_policy_parses_stream_and_stop_reason():
    def ch(i, text, finish=None, stop_reason=None):
        return {"choices": [{"index": i, "text": text, "finish_reason": finish, "stop_reason": stop_reason}]}
    chunks = [ch(0, "Step"), ch(1, "Ans"), ch(0, " one", "stop", "\n\n"), ch(1, "wer", "stop", None)]
    pol = VllmPolicy("http://vllm", "m", tokenizer=None, transport=_sse(chunks))
    evs = [e async for e in pol.stream("p", 2, ["\n\n"], 10, 0.7, 0.8, 20)]
    dones = {e["index"]: e for e in evs if e["type"] == "done"}
    assert dones[0]["text"] == "Step one" and dones[0]["finished"] is False  # hit the step boundary
    assert dones[1]["text"] == "Answer" and dones[1]["finished"] is True     # EOS
    assert [e["text"] for e in evs if e["type"] == "delta" and e["index"] == 0] == ["Step", " one"]


async def test_vllm_http_error_includes_body():
    t = httpx.MockTransport(lambda req: httpx.Response(400, text="maximum context length is 12288 tokens"))
    pol = VllmPolicy("http://vllm", "m", transport=t)
    with pytest.raises(RuntimeError, match="maximum context length"):
        async for _ in pol.stream("p", 1, None, 10, 0.7, 0.8, 20):
            pass


def test_generate_without_deltas_sends_only_dones():
    client, _ = make()
    with client.websocket_connect("/ws") as ws:
        auth(ws)
        ws.send_json({"op": "generate", "job": "q", "prompt": "p", "n": 2, "stop": None, "max_tokens": 10,
                      "temperature": 0.7, "top_p": 0.8, "top_k": 20, "deltas": False})
        msgs = until_end(ws, "q")
        assert [m["type"] for m in msgs] == ["done", "done", "end"]
        assert msgs[0]["text"] == "w0 w1 "


class FakeTok:
    """Whitespace tokenizer: <extra_0> -> 7, pad -> 0, other words -> 3..7 by length."""
    padding_side = "right"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        return " ".join(m["content"] for m in msgs)

    def encode(self, text, return_tensors=None):
        return [7]

    def __call__(self, texts, return_tensors="pt", padding=True):
        import torch
        rows = [[7 if w == "<extra_0>" else 3 + len(w) % 4 for w in t.replace("<extra_0>", " <extra_0> ").split()]
                for t in texts]
        width = max(len(r) for r in rows)
        ids = torch.tensor([r + [0] * (width - len(r)) for r in rows])
        mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows])
        from types import SimpleNamespace
        return SimpleNamespace(input_ids=ids, attention_mask=mask)


class FakePrmModel:
    device = "cpu"

    def __call__(self, input_ids, attention_mask=None):
        import torch
        t = torch.arange(input_ids.shape[1]).expand_as(input_ids).float()
        pos = 0.05 * t + 0.02 * input_ids.float()          # depends only on token + position
        return (torch.stack([torch.zeros_like(pos), pos], dim=-1),)


def test_qwen_prm_batches_without_changing_scores():
    pytest.importorskip("torch")
    from colab.gateway import QwenPrm
    prm = QwenPrm.from_parts(FakeTok(), FakePrmModel(), batch_size=2)
    sols = [["a b c", "dd"], ["e"], ["fff g", "h", "iiii jj kk"]]
    batched = prm.score("problem text", sols)
    singles = [prm.score("problem text", [s])[0] for s in sols]
    assert [len(x) for x in batched] == [2, 1, 3]
    assert batched == singles
    assert all(0 <= v <= 1 for row in batched for v in row)
