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
