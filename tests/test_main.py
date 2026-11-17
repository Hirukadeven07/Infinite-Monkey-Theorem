import socket
import time

import pytest
from fastapi.testclient import TestClient

from tests.helpers import FIXTURE, QS

QID = QS[0].id


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TTS_DATA", str(FIXTURE))
    monkeypatch.setenv("TTS_DB", str(tmp_path / "t.db"))
    monkeypatch.setenv("TTS_MOCK_DELAY", "0")
    from app.main import app
    with TestClient(app) as c:
        yield c


def collect_until(ws, stop_type, limit=5000):
    types = []
    for _ in range(limit):
        msg = ws.receive_json()
        types.append(msg["type"])
        if msg["type"] in (stop_type, "error"):
            return types, msg
    raise AssertionError("no terminal event")


def test_backend_default_is_mock(client):
    r = client.get("/api/backend").json()
    assert r["kind"] == "mock" and r["connected"] is True and r["info"]["mock"] is True


def test_questions_filter(client):
    qs = client.get("/api/questions").json()
    assert len(qs) == 10 and {"id", "level", "subject", "problem", "answer"} <= set(qs[0])
    assert {q["level"] for q in client.get("/api/questions?level=2").json()} == {2}


def test_live_ws_bon_runs(client):
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"op": "run", "question_id": QID, "method": "bon", "budget": 4})
        types, last = collect_until(ws, "run_end")
        assert "sample_delta" in types and "tally" in types and last["type"] == "run_end"
    runs = client.get("/api/runs?kind=live").json()
    assert len(runs) == 1 and runs[0]["config"]["method"] == "bon"


def test_live_custom_question(client):
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"op": "run", "custom": {"problem": "What is 6*7?", "answer": "42"},
                      "method": "majority", "budget": 4})
        types, last = collect_until(ws, "run_end")
        assert last["type"] == "run_end"


@pytest.mark.parametrize("msg", [
    {"op": "run", "question_id": QID, "method": "beam", "budget": 6},
    {"op": "run", "question_id": "nope", "method": "bon", "budget": 4},
    {"op": "run", "question_id": QID, "method": "magic", "budget": 4},
])
def test_live_bad_requests_send_error(client, msg):
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json(msg)
        types, last = collect_until(ws, "run_end")
        assert last["type"] == "error" and last["message"]


def test_live_stop(client):
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"op": "run", "question_id": QID, "method": "revisions", "budget": 8})
        assert ws.receive_json()["type"] == "run_start"
        ws.send_json({"op": "stop"})
        types, last = collect_until(ws, "stopped")
        assert last["type"] == "stopped" and "run_end" not in types


def test_backend_bad_url_502(client):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    r = client.post("/api/backend", json={"kind": "colab", "url": f"http://127.0.0.1:{port}", "token": "x"})
    assert r.status_code == 502 and r.json()["detail"]
    assert client.get("/api/backend").json()["kind"] == "mock"


def test_bench_metrics_and_export(client):
    cfg = {"per_level": 1, "pool_size": 4, "beam_budgets": [4], "revision_budgets": [2], "repeats": 10}
    rid = client.post("/api/bench", json=cfg).json()["run_id"]
    for _ in range(300):
        m = client.get(f"/api/bench/{rid}/metrics").json()
        if m["progress"]["status"] != "running":
            break
        time.sleep(0.05)
    assert m["progress"]["status"] == "done" and m["progress"]["done"] == 15
    assert m["mock"] is True and m["baseline"] is not None
    assert {"majority", "bon", "beam", "revisions", "pass@1"} <= set(m["curves"])
    assert m["compute_optimal"] and m["by_level"] and m["cost"]["bon"]
    csv = client.get(f"/api/bench/{rid}/export.csv").text.splitlines()
    assert csv[0] == "run_id,question_id,level,method,budget,accuracy,tokens,seconds" and len(csv) > 15
    js = client.get(f"/api/bench/{rid}/export.json").json()
    assert js["config"]["pool_size"] == 4 and js["rows"]
    assert client.post("/api/bench", json={"beam_budgets": [6]}).status_code == 422
    assert client.post("/api/bench/999/resume").status_code == 404
