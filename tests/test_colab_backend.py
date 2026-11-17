import asyncio
import socket
import threading
import time

import pytest
import uvicorn

from app.backends import ColabBackend, ws_url
from app.schema import BackendError, Delta, Done, Sampling
from colab.gateway import create_app
from tests.fakes import INFO, FakePolicy, FakePrm

TOKEN = "t0k"


@pytest.mark.parametrize("raw", ["https://a.trycloudflare.com", "https://a.trycloudflare.com/",
                                 "https://a.trycloudflare.com/ws", " a.trycloudflare.com "])
def test_ws_url_normalizes(raw):  # Review Focus 4
    assert ws_url(raw) == "wss://a.trycloudflare.com/ws"


def test_ws_url_http():
    assert ws_url("http://127.0.0.1:9000") == "ws://127.0.0.1:9000/ws"


class Gateway:
    def __init__(self, slow: bool):
        self.policy = FakePolicy(slow=slow)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        config = uvicorn.Config(create_app(self.policy, FakePrm(), TOKEN, INFO), host="127.0.0.1",
                                port=self.port, log_level="warning", timeout_graceful_shutdown=1)
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started and time.time() < deadline:
            time.sleep(0.02)
        self.url = f"http://127.0.0.1:{self.port}"

    def stop(self):
        self.server.should_exit = True
        self.thread.join(5)


@pytest.fixture
def gateway(request):
    gw = Gateway(slow=getattr(request, "param", False))
    yield gw
    gw.stop()


async def test_generate_and_score_against_live_gateway(gateway):
    be = ColabBackend(gateway.url, TOKEN)
    info = await be.connect()
    assert info == INFO and be.info == INFO and be.connected
    evs = [e async for e in be.generate("p", 2, Sampling(), stop=["\n\n"])]
    dones = [e for e in evs if isinstance(e, Done)]
    assert sorted(d.index for d in dones) == [0, 1] and all(d.finished is False for d in dones)
    assert any(isinstance(e, Delta) for e in evs)
    assert gateway.policy.calls[0]["stop"] == ["\n\n"]
    assert await be.score("p", [["a"], ["b", "c"]]) == [[0.5], [0.5, 0.5]]
    # concurrent jobs share one socket
    a, b = await asyncio.gather(be.score("p", [["x"]]), be.score("p", [["y", "z"]]))
    assert a == [[0.5]] and b == [[0.5, 0.5]]
    await be.close()
    assert not be.connected


async def test_bad_token(gateway):
    be = ColabBackend(gateway.url, "wrong")
    with pytest.raises(BackendError, match="token"):
        await be.connect()


async def test_unreachable_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(BackendError):
        await ColabBackend(f"http://127.0.0.1:{port}", TOKEN).connect()


@pytest.mark.parametrize("gateway", [True], indirect=True)
async def test_disconnect_raises(gateway):  # Review Focus 2
    be = ColabBackend(gateway.url, TOKEN)
    await be.connect()
    with pytest.raises(BackendError):
        async for ev in be.generate("p", 1, Sampling()):
            if isinstance(ev, Delta):
                await asyncio.to_thread(gateway.stop)
    assert not be.connected
    with pytest.raises(BackendError):
        await be.score("p", [["a"]])


@pytest.mark.parametrize("gateway", [True], indirect=True)
async def test_early_close_sends_cancel(gateway):  # Review Focus 3
    be = ColabBackend(gateway.url, TOKEN)
    await be.connect()
    gen = be.generate("p", 1, Sampling())
    first = await gen.__anext__()
    assert isinstance(first, Delta)
    await gen.aclose()
    for _ in range(100):
        if gateway.policy.cancelled:
            break
        await asyncio.sleep(0.02)
    assert gateway.policy.cancelled
    await be.close()
