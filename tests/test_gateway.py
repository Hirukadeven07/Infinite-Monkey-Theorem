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
