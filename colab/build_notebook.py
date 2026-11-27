"""Generate colab/tts_server.ipynb from colab/gateway.py.

    python colab/build_notebook.py          # write the notebook
    python colab/build_notebook.py --check  # exit 1 if the notebook is out of date

The gateway lives in a real .py file so it can be tested locally; the notebook embeds it
verbatim with %%writefile so Colab needs nothing but this one file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "tts_server.ipynb"
VLLM_VERSION = "0.34.0"  # latest release on PyPI when this was written (2026-11-17)

INTRO = r"""# Infinite Monkey: inference server

Hosts the models behind the Infinite Monkey web app on a Colab GPU and exposes them through a
Cloudflare quick tunnel.

| Component | Default | Runtime |
|---|---|---|
| Policy | `Qwen/Qwen3-1.7B` | vLLM @VLLM_VERSION@, OpenAI-compatible completions |
| Process reward model | `Qwen/Qwen2.5-Math-PRM-7B` | Transformers; bf16 on 40 GB+ GPUs, 4-bit otherwise |
| Gateway | `colab/gateway.py` | FastAPI WebSocket with token auth |

### Usage

1. *Runtime > Change runtime type* and pick a GPU. A100 or L4 recommended; T4 works but is slow.
2. *Runtime > Run all.* The first run downloads about 20 GB of weights (5 to 15 minutes).
3. Paste the printed endpoint and token into the app's backend dialog.

### Notes

- Each cell depends on the ones above it. After a restart, use *Run all*.
- If the install cell asks for a restart, restart the session and *Run all* again; the install is skipped.
- The endpoint changes every session. Benchmark runs resume after you reconnect.
- Anyone with both the endpoint and the token can use this GPU. Keep them private.
""".replace("@VLLM_VERSION@", VLLM_VERSION)

SETUP = r"""## Setup

`Qwen/Qwen3-4B` needs an L4 or A100. `MAX_MODEL_LEN` leaves room for revision prompts that carry
four previous attempts.
"""

CONFIG = r"""# @title Configuration
POLICY_MODEL = "Qwen/Qwen3-1.7B"  # @param ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"]
PRM_MODEL = "Qwen/Qwen2.5-Math-PRM-7B"
MAX_MODEL_LEN = 12288
VLLM_PORT = 8001
GATEWAY_PORT = 9000
"""

INSTALL = r"""# @title Install dependencies
import importlib.metadata
import subprocess
import sys

VLLM_VERSION = "@VLLM_VERSION@"
EXTRA_PACKAGES = ["bitsandbytes", "fastapi", "uvicorn", "httpx"]
# vLLM may upgrade these. A copy already imported by this kernel would then be stale.
SHARED_PACKAGES = ("torch", "transformers", "numpy", "triton")


def installed_version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist).split("+")[0]
    except importlib.metadata.PackageNotFoundError:
        return None


if installed_version("vllm") == VLLM_VERSION:
    print(f"vllm=={VLLM_VERSION} already installed, continuing.")
else:
    before = {pkg: installed_version(pkg) for pkg in SHARED_PACKAGES}
    proc = subprocess.Popen(
        [sys.executable, "-m", "pip", "install", f"vllm=={VLLM_VERSION}", *EXTRA_PACKAGES],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for line in proc.stdout:
        print(line, end="")
    if proc.wait() != 0:
        raise RuntimeError("pip install failed; see the output above.")

    stale = [pkg for pkg in SHARED_PACKAGES if pkg in sys.modules and installed_version(pkg) != before[pkg]]
    if stale:
        raise RuntimeError(
            f"{', '.join(stale)} changed after this kernel imported it. "
            "Restart the session, then Run all (the install will be skipped)."
        )
    print(f"Installed vllm=={VLLM_VERSION}, continuing.")
""".replace("@VLLM_VERSION@", VLLM_VERSION)

HARDWARE = r"""## Hardware

Precision and the GPU memory split between vLLM and the reward model follow from the detected GPU.
"""

GPU = r"""# @title Detect GPU
if "POLICY_MODEL" not in globals():
    raise RuntimeError("Configuration not loaded. Use Runtime > Run all.")

import torch

if not torch.cuda.is_available():
    raise RuntimeError("No GPU found. Select one under Runtime > Change runtime type.")

props = torch.cuda.get_device_properties(0)
GPU_NAME = props.name
GPU_MEMORY_GB = props.total_memory / 1024**3

# Pre-Ampere GPUs (T4) lack bf16.
DTYPE = "bfloat16" if torch.cuda.get_device_capability(0) >= (8, 0) else "half"
# Full-precision PRM only when the GPU fits it next to the policy.
PRM_QUANT = "bf16" if GPU_MEMORY_GB >= 40 else "4bit"
# vLLM's share of GPU memory. Small GPUs need a larger share to fit the policy's KV cache.
GPU_UTIL = 0.35 if GPU_MEMORY_GB >= 40 else 0.30 if GPU_MEMORY_GB >= 20 else 0.45

print(f"{GPU_NAME} | {GPU_MEMORY_GB:.0f} GB | policy {DTYPE} | PRM {PRM_QUANT} | vLLM memory share {GPU_UTIL:.2f}")
"""

GATEWAY_MD = r"""## Gateway

Generated from `colab/gateway.py`. Edit that file and run `python colab/build_notebook.py` rather
than changing this cell.
"""

POLICY_MD = r"""## Policy server

Starts vLLM in the background, or reuses one that survived a kernel restart. Logs go to `vllm.log`.
"""

VLLM = r"""# @title Start vLLM
if "DTYPE" not in globals():
    raise RuntimeError("GPU settings not loaded. Use Runtime > Run all.")

import subprocess
import time

import httpx

VLLM_URL = f"http://127.0.0.1:{VLLM_PORT}"
VLLM_STARTUP_TIMEOUT_S = 900


def vllm_healthy() -> bool:
    try:
        return httpx.get(f"{VLLM_URL}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def log_tail(path: str, n_chars: int = 4000) -> str:
    with open(path) as f:
        return f.read()[-n_chars:]


if vllm_healthy():
    served = httpx.get(f"{VLLM_URL}/v1/models", timeout=5).json()["data"][0]["id"]
    if served != POLICY_MODEL:
        raise RuntimeError(
            f"Port {VLLM_PORT} is serving {served}, not {POLICY_MODEL}. "
            "Use Runtime > Disconnect and delete runtime, then Run all."
        )
    print(f"Reusing the running vLLM server ({served}).")
else:
    vllm_log = open("vllm.log", "w")
    vllm_proc = subprocess.Popen(
        [
            "vllm", "serve", POLICY_MODEL,
            "--port", str(VLLM_PORT),
            "--dtype", DTYPE,
            "--gpu-memory-utilization", str(GPU_UTIL),
            "--max-model-len", str(MAX_MODEL_LEN),
        ],
        stdout=vllm_log, stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + VLLM_STARTUP_TIMEOUT_S
    while not vllm_healthy():
        if vllm_proc.poll() is not None:
            print(log_tail("vllm.log"))
            raise RuntimeError("vLLM exited during startup; see the log above.")
        if time.monotonic() > deadline:
            print(log_tail("vllm.log"))
            raise RuntimeError(f"vLLM was not healthy after {VLLM_STARTUP_TIMEOUT_S} s; see the log above.")
        time.sleep(5)
    print(f"vLLM serving {POLICY_MODEL} at {VLLM_URL}.")

VLLM_READY = True
"""

PRM_MD = r"""## Reward model and gateway

Loads the PRM, then serves the authenticated gateway on `GATEWAY_PORT`. Re-running this cell
replaces the previous gateway and issues a new token.
"""

GATEWAY = r"""# @title Load PRM and start gateway
if "VLLM_READY" not in globals():
    raise RuntimeError("vLLM is not running. Use Runtime > Run all.")

import importlib
import importlib.metadata
import secrets
import threading
import time

import uvicorn
from transformers import AutoTokenizer

import gateway

importlib.reload(gateway)  # pick up a rewritten gateway.py on re-run
from gateway import QwenPrm, VllmPolicy, create_app

if "server" in globals() and server.started:
    server.should_exit = True
    time.sleep(1.5)

prm = QwenPrm(PRM_MODEL, PRM_QUANT)
policy = VllmPolicy(VLLM_URL, POLICY_MODEL, tokenizer=AutoTokenizer.from_pretrained(POLICY_MODEL))

TOKEN = secrets.token_urlsafe(24)
INFO = {
    "policy_model": POLICY_MODEL,
    "policy_dtype": DTYPE,
    "prm_model": PRM_MODEL,
    "prm_precision": PRM_QUANT,
    "gpu": GPU_NAME,
    "vllm_version": importlib.metadata.version("vllm"),
    "mock": False,
}

server = uvicorn.Server(
    uvicorn.Config(create_app(policy, prm, TOKEN, INFO), host="127.0.0.1", port=GATEWAY_PORT, log_level="warning")
)
server_thread = threading.Thread(target=server.run, daemon=True)
server_thread.start()
while not server.started:
    if not server_thread.is_alive():
        raise RuntimeError(f"Gateway failed to start on port {GATEWAY_PORT}.")
    time.sleep(0.2)

for key, value in INFO.items():
    print(f"{key:<14} {value}")
"""

TUNNEL_MD = r"""## Tunnel

Opens a Cloudflare quick tunnel to the gateway and prints the endpoint and token for the app.
"""

TUNNEL = r"""# @title Open tunnel
if "TOKEN" not in globals():
    raise RuntimeError("Gateway is not running. Use Runtime > Run all.")

import os
import queue
import re
import subprocess
import threading
import time
import urllib.request

CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
TUNNEL_TIMEOUT_S = 90

if not os.path.exists("cloudflared"):
    urllib.request.urlretrieve(CLOUDFLARED_URL, "cloudflared")
    os.chmod("cloudflared", 0o755)

if "tunnel" in globals() and tunnel.poll() is None:
    tunnel.terminate()

tunnel = subprocess.Popen(
    ["./cloudflared", "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{GATEWAY_PORT}"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)


def drain(stream, sink):
    for line in stream:
        sink.put(line)


# cloudflared blocks if its output pipe fills, so keep reading for the life of the process.
tunnel_output = queue.Queue()
threading.Thread(target=drain, args=(tunnel.stdout, tunnel_output), daemon=True).start()

PUBLIC_URL = None
log_lines = []
deadline = time.monotonic() + TUNNEL_TIMEOUT_S
while PUBLIC_URL is None and time.monotonic() < deadline and tunnel.poll() is None:
    try:
        line = tunnel_output.get(timeout=1)
    except queue.Empty:
        continue
    log_lines.append(line.rstrip())
    if match := re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line):
        PUBLIC_URL = match.group(0)

if PUBLIC_URL is None:
    print("\n".join(log_lines[-30:]))
    raise RuntimeError(f"cloudflared printed no tunnel URL within {TUNNEL_TIMEOUT_S} s; see the output above. Re-run this cell.")

print(f"Endpoint  {PUBLIC_URL}")
print(f"Token     {TOKEN}")
print("\nPaste both into the app's backend dialog. The tunnel can take about 30 s to start answering.")
"""

MONITOR_MD = r"""## Monitor

Optional. Reports health once a minute; interrupting this cell leaves the servers running.
"""

MONITOR = r"""# @title Monitor health
import time

import httpx

while True:
    stamp = time.strftime("%H:%M:%S")
    try:
        r = httpx.get(f"http://127.0.0.1:{GATEWAY_PORT}/health", headers={"Authorization": f"Bearer {TOKEN}"}, timeout=5)
        vllm_state = "ok" if vllm_healthy() else "down"
        tunnel_state = "ok" if tunnel.poll() is None else "exited"
        print(f"{stamp}  gateway {r.status_code}  vllm {vllm_state}  tunnel {tunnel_state}")
    except httpx.HTTPError as e:
        print(f"{stamp}  health check failed: {e}")
    time.sleep(60)
"""


def _source(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def _code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _source(text)}


def build() -> str:
    gateway_src = (HERE / "gateway.py").read_text(encoding="utf-8").strip()
    cells = [
        _md(INTRO),
        _md(SETUP), _code(CONFIG), _code(INSTALL),
        _md(HARDWARE), _code(GPU),
        _md(GATEWAY_MD), _code("%%writefile gateway.py\n" + gateway_src),
        _md(POLICY_MD), _code(VLLM),
        _md(PRM_MD), _code(GATEWAY),
        _md(TUNNEL_MD), _code(TUNNEL),
        _md(MONITOR_MD), _code(MONITOR),
    ]
    nb = {
        "cells": cells,
        "metadata": {"accelerator": "GPU",
                     "colab": {"name": "tts_server.ipynb", "provenance": [], "gpuType": "L4", "toc_visible": True},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 0,
    }
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    content = build()
    if "--check" in sys.argv:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != content:
            print("colab/tts_server.ipynb is out of date; run: python colab/build_notebook.py")
            return 1
        return 0
    OUT.write_text(content, encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
