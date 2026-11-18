import ast
import json
from pathlib import Path

NB = Path("colab/tts_server.ipynb")


def load():
    return json.loads(NB.read_text(encoding="utf-8"))


def test_notebook_embeds_current_gateway():
    nb = load()
    src = "".join("".join(c["source"]) for c in nb["cells"])
    assert Path("colab/gateway.py").read_text(encoding="utf-8").strip() in src
    assert "trycloudflare" in src and "Qwen/Qwen3-1.7B" in src and "Qwen/Qwen2.5-Math-PRM-7B" in src
    assert nb["nbformat"] == 4
    assert nb["metadata"]["accelerator"] == "GPU"


def test_notebook_code_cells_are_valid_python():
    for cell in load()["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = "".join(cell["source"]).splitlines()
        if lines and lines[0].startswith("%%writefile"):
            body = "\n".join(lines[1:])
        else:
            body = "\n".join(l for l in lines if not l.lstrip().startswith(("!", "%")))
        ast.parse(body)


def test_notebook_is_up_to_date():
    import subprocess
    import sys
    before = NB.read_text(encoding="utf-8")
    subprocess.run([sys.executable, "colab/build_notebook.py", "--check"], check=True)
    assert NB.read_text(encoding="utf-8") == before


def _code_cells():
    return ["".join(c["source"]) for c in load()["cells"] if c["cell_type"] == "code"]


def _cell_with(marker):
    return next(c for c in _code_cells() if marker in c)


def test_install_cell_skips_when_installed_and_restarts_after_installing():
    cell = _cell_with("pip")
    assert 'importlib.metadata.version("vllm")' in cell     # already installed -> skip
    assert "os.kill(os.getpid()" in cell                    # fresh install -> restart so new torch loads
    assert "Run all" in cell                                 # and tell the user what to do next
    assert not cell.lstrip().startswith("!pip")


GUARDED = {
    "get_device_properties": "POLICY_MODEL",   # GPU check needs the config cell
    '"vllm", "serve"': "DTYPE",                 # vLLM needs the GPU check
    "QwenPrm(PRM_MODEL": "vllm_proc",           # gateway needs vLLM running
    "--no-autoupdate": "TOKEN",                 # tunnel needs the gateway
}


def test_each_stage_checks_its_prerequisites():
    for marker, needed in GUARDED.items():
        guard = _cell_with(marker).split(marker)[0]
        assert needed in guard and "not in globals()" in guard, f"{marker!r} cell must check for {needed} first"


def test_guards_work_in_a_fresh_session():
    # The reported failure: after a restart nothing above has run. Each guard must then raise a
    # clear RuntimeError (pointing at Run all), not a NameError - including for a helper.
    import pytest
    for marker in GUARDED:
        guard = _cell_with(marker).split(marker)[0]
        guard = guard[:guard.index("import ")] if "import " in guard else guard
        with pytest.raises(RuntimeError, match="Run all"):
            exec(guard, {})
