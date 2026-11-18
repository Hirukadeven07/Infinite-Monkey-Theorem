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

INTRO = """# Test-time scaling server (Colab GPU)

This notebook serves **Qwen3** (policy, via vLLM) and **Qwen2.5-Math-PRM-7B** (process reward model)
to the local *Infinite Monkey* website through a temporary Cloudflare tunnel.

1. **Runtime → Change runtime type → GPU.** An **A100** or **L4** is recommended; a T4 works but is slow.
2. **Runtime → Run all.** The first run installs vLLM and downloads ~20 GB of weights (5–15 min).
3. Copy the **URL** and **TOKEN** printed at the end into the website's backend dialog
   (click the backend pill in the top-right corner).

Keep this tab open while you use the site. The URL changes every session; benchmark runs can be
resumed after reconnecting. Anyone with both the URL *and* the token can use this GPU, so don't share them.
"""

CONFIG = """# --- configuration --------------------------------------------------------------
POLICY_MODEL = "Qwen/Qwen3-1.7B"        # also try "Qwen/Qwen3-0.6B"; "Qwen/Qwen3-4B" needs an A100/L4
PRM_MODEL = "Qwen/Qwen2.5-Math-PRM-7B"
VLLM_PORT = 8001
GATEWAY_PORT = 9000
MAX_MODEL_LEN = 12288                     # room for revision prompts with 4 previous attempts
"""

INSTALL = f"""# vLLM pinned to the version this notebook was written against.
!pip install -q vllm=={VLLM_VERSION} bitsandbytes fastapi uvicorn httpx
"""

GPU = """import torch

assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> GPU"
props = torch.cuda.get_device_properties(0)
GPU_NAME = props.name
TOTAL_GB = props.total_memory / 1024**3
DTYPE = "half" if torch.cuda.get_device_capability(0) < (8, 0) else "bfloat16"   # T4 has no bf16
PRM_QUANT = "bf16" if TOTAL_GB >= 40 else "4bit"
GPU_UTIL = 0.35 if TOTAL_GB >= 40 else (0.30 if TOTAL_GB >= 20 else 0.45)
print(f"GPU: {GPU_NAME} ({TOTAL_GB:.0f} GB) | policy dtype {DTYPE} | vLLM share {GPU_UTIL} | PRM {PRM_QUANT}")
"""

VLLM = """import subprocess, sys, time, httpx

vllm_log = open("vllm.log", "w")
vllm_proc = subprocess.Popen(
    ["vllm", "serve", POLICY_MODEL, "--port", str(VLLM_PORT), "--dtype", DTYPE,
     "--gpu-memory-utilization", str(GPU_UTIL), "--max-model-len", str(MAX_MODEL_LEN)],
    stdout=vllm_log, stderr=subprocess.STDOUT)

deadline = time.time() + 900
while time.time() < deadline:
    if vllm_proc.poll() is not None:
        print(open("vllm.log").read()[-4000:])
        raise RuntimeError("vLLM exited during startup (log above)")
    try:
        if httpx.get(f"http://127.0.0.1:{VLLM_PORT}/health", timeout=2).status_code == 200:
            break
    except httpx.HTTPError:
        pass
    time.sleep(5)
else:
    print(open("vllm.log").read()[-4000:])
    raise RuntimeError("vLLM did not become healthy within 15 minutes")
print("vLLM is serving", POLICY_MODEL)
"""

GATEWAY = """import secrets, threading, importlib.metadata
import uvicorn
from transformers import AutoTokenizer
from gateway import QwenPrm, VllmPolicy, create_app

prm = QwenPrm(PRM_MODEL, PRM_QUANT)
policy = VllmPolicy(f"http://127.0.0.1:{VLLM_PORT}", POLICY_MODEL, tokenizer=AutoTokenizer.from_pretrained(POLICY_MODEL))
TOKEN = secrets.token_urlsafe(24)
INFO = {"policy_model": POLICY_MODEL, "policy_dtype": DTYPE, "prm_model": PRM_MODEL, "prm_precision": PRM_QUANT,
        "gpu": GPU_NAME, "vllm_version": importlib.metadata.version("vllm"), "mock": False}

server = uvicorn.Server(uvicorn.Config(create_app(policy, prm, TOKEN, INFO), host="127.0.0.1",
                                       port=GATEWAY_PORT, log_level="warning"))
threading.Thread(target=server.run, daemon=True).start()
while not server.started:
    time.sleep(0.2)
print("Gateway up:", INFO)
"""

TUNNEL = """import os, re, threading, urllib.request

if not os.path.exists("cloudflared"):
    urllib.request.urlretrieve(
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "cloudflared")
    os.chmod("cloudflared", 0o755)

tunnel = subprocess.Popen(["./cloudflared", "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{GATEWAY_PORT}"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
PUBLIC_URL = None
for line in tunnel.stdout:
    m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", line)
    if m:
        PUBLIC_URL = m.group(0)
        break
threading.Thread(target=lambda: [None for _ in tunnel.stdout], daemon=True).start()  # keep draining the pipe

from IPython.display import HTML, display
display(HTML(f'''
<div style="font:16px monospace;padding:14px;border:2px solid #2a78d6;border-radius:10px">
<div>URL&nbsp;&nbsp;&nbsp;<b>{PUBLIC_URL}</b></div>
<div>TOKEN&nbsp;<b>{TOKEN}</b></div>
<div style="font-size:12px;opacity:.7;margin-top:6px">Paste both into the website's backend dialog. The tunnel may need ~30 s before it answers.</div>
</div>'''))
"""

KEEPALIVE = """# Optional: keep this cell running to watch health (stop it any time; the server keeps running).
while True:
    try:
        r = httpx.get(f"http://127.0.0.1:{GATEWAY_PORT}/health", headers={"Authorization": f"Bearer {TOKEN}"}, timeout=5)
        print(time.strftime("%H:%M:%S"), "gateway", r.status_code, "| vLLM alive:", vllm_proc.poll() is None)
    except Exception as e:
        print(time.strftime("%H:%M:%S"), "health check failed:", e)
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
        _code(CONFIG),
        _code(INSTALL),
        _md("## 1. GPU check"), _code(GPU),
        _md("## 2. Gateway code (generated from `colab/gateway.py` in the repo)"),
        _code("%%writefile gateway.py\n" + gateway_src),
        _md("## 3. Start vLLM (policy model)"), _code(VLLM),
        _md("## 4. Load the PRM and start the gateway"), _code(GATEWAY),
        _md("## 5. Open the tunnel and print the URL + token"), _code(TUNNEL),
        _md("## 6. Health (optional)"), _code(KEEPALIVE),
    ]
    nb = {
        "cells": cells,
        "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
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
