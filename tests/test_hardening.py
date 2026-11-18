"""Gateway input hardening, cross-site protection for the local app, and a loud tunnel failure."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from colab.gateway import create_app
from tests.fakes import INFO, FakePolicy, FakePrm
from tests.helpers import FIXTURE, QS


def gateway():
    return TestClient(create_app(FakePolicy(), FakePrm(), "t", INFO))


def test_gateway_hides_api_docs():
    c = gateway()
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert c.get(path).status_code == 404


@pytest.mark.parametrize("first", ['{"op": "auth", "token": 123}', "not json at all", '["auth"]'])
def test_gateway_bad_auth_messages_close_4401(first):
    with gateway().websocket_connect("/ws") as ws:
        ws.send_text(first)
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
        assert e.value.code == 4401


def test_gateway_malformed_message_does_not_drop_socket():
    with gateway().websocket_connect("/ws") as ws:
        ws.send_json({"op": "auth", "token": "t"})
        ws.receive_json()
        ws.send_text("{broken")
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"op": "score", "job": "s", "problem": "p", "solutions": [["a"]]})
        assert ws.receive_json() == {"job": "s", "type": "scores", "scores": [[0.5]]}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TTS_DATA", str(FIXTURE))
    monkeypatch.setenv("TTS_DB", str(tmp_path / "t.db"))
    monkeypatch.setenv("TTS_MOCK_DELAY", "0")
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_cross_site_post_is_refused(client):
    r = client.post("/api/backend", json={"kind": "mock"}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    ok = client.post("/api/backend", json={"kind": "mock"}, headers={"Origin": "http://testserver"})
    assert ok.status_code == 200
    assert client.post("/api/backend", json={"kind": "mock"}).status_code == 200  # non-browser clients


def test_cross_site_live_socket_is_refused(client):
    with pytest.raises(WebSocketDisconnect) as e:
        with client.websocket_connect("/ws/live", headers={"Origin": "https://evil.example"}) as ws:
            ws.send_json({"op": "run", "question_id": QS[0].id, "method": "bon", "budget": 2})
            ws.receive_json()
    assert e.value.code == 4403
    with client.websocket_connect("/ws/live", headers={"Origin": "http://testserver"}) as ws:
        ws.send_json({"op": "run", "question_id": QS[0].id, "method": "bon", "budget": 2})
        assert ws.receive_json()["type"] == "run_start"


def test_tunnel_cell_fails_loudly_without_url():
    nb = json.loads(Path("colab/tts_server.ipynb").read_text(encoding="utf-8"))
    cell = next("".join(c["source"]) for c in nb["cells"] if "trycloudflare" in "".join(c["source"])
                and c["cell_type"] == "code")
    assert "if PUBLIC_URL is None" in cell and "raise RuntimeError" in cell
