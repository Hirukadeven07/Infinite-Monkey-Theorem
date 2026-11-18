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


class FakePip:
    """Stands in for subprocess.Popen running pip: prints lines, returns a code, optionally
    'upgrades' packages by flipping the fake installed versions."""

    def __init__(self, versions, rc=0, after=None):
        self.versions, self.rc, self.after, self.calls = versions, rc, after or {}, []

    def __call__(self, args, **kw):
        self.calls.append(args)
        outer = self

        class P:
            stdout = iter(["Collecting vllm\n", "Successfully installed vllm\n"])

            def wait(self):
                outer.versions.update(outer.after)
                return outer.rc
        return P()


def run_install_cell(monkeypatch, versions, rc=0, after=None, loaded=()):
    import importlib.metadata
    import os
    import subprocess
    import sys
    pip = FakePip(versions, rc, after)
    monkeypatch.setattr(subprocess, "Popen", pip)

    def version(name):
        if versions.get(name) is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]
    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(os, "kill", lambda *a: (_ for _ in ()).throw(AssertionError("must not kill the kernel")))
    for m in ("torch", "transformers", "numpy", "triton"):  # simulate a fresh session...
        monkeypatch.delitem(sys.modules, m, raising=False)
    for m in loaded:                                         # ...except what the test says is loaded
        monkeypatch.setitem(sys.modules, m, object())
    exec(_cell_with("pip"), {})
    return pip


def test_install_skipped_when_pinned_vllm_present(monkeypatch, capsys):
    pip = run_install_cell(monkeypatch, {"vllm": "0.34.0"})
    assert pip.calls == [] and "already installed" in capsys.readouterr().out


def test_fresh_install_continues_without_restart(monkeypatch, capsys):
    pip = run_install_cell(monkeypatch, {"vllm": None, "torch": "2.5.0"}, after={"vllm": "0.34.0", "torch": "2.9.0"})
    out = capsys.readouterr().out
    assert len(pip.calls) == 1 and "Successfully installed vllm" in out   # pip output is shown
    assert "continuing" in out                                             # torch wasn't loaded yet


def test_failed_pip_raises_with_message(monkeypatch):
    import pytest
    with pytest.raises(RuntimeError, match="pip install failed"):
        run_install_cell(monkeypatch, {"vllm": None}, rc=1)


def test_restart_requested_only_if_a_loaded_package_changed(monkeypatch):
    import pytest
    with pytest.raises(RuntimeError, match="[Rr]estart"):
        run_install_cell(monkeypatch, {"vllm": None, "torch": "2.5.0"},
                         after={"vllm": "0.34.0", "torch": "2.9.0"}, loaded=("torch",))


GUARDED = {
    "get_device_properties": "POLICY_MODEL",   # GPU check needs the config cell
    '"vllm", "serve"': "DTYPE",                 # vLLM needs the GPU check
    "QwenPrm(PRM_MODEL": "VLLM_READY",          # gateway needs vLLM running
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


def test_gateway_cell_reloads_edited_gateway_module():
    # Re-running after gateway.py is rewritten must not keep the stale imported module.
    cell = _cell_with("QwenPrm(PRM_MODEL")
    assert "importlib.reload(gateway)" in cell
    assert cell.index("importlib.reload(gateway)") < cell.index("from gateway import")


def test_vllm_cell_reuses_a_server_already_on_the_port():
    # After a kernel restart the old vLLM process can still hold the port and GPU memory.
    cell = _cell_with('"vllm", "serve"')
    assert cell.index("/health") < cell.index("subprocess.Popen")
    assert "VLLM_READY = True" in cell
