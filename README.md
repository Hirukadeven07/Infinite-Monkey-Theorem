<div align="center">

# Infinite Monkey Theorem

<img src="assets/monkey-typewriter.png" alt="A chimpanzee seated at a typewriter" width="420" />

<br/>

![Status](https://img.shields.io/badge/status-in%20development-orange?style=flat-square)
![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)

</div>

## About

This project aims to study the effect of **inference scaling**: how spending more compute at inference time (test-time scaling) changes a model's performance, rather than only scaling model size or training compute.

It ships a **test-time scaling visualizer**: a local website where you watch a small model (Qwen3) answer a MATH-500 problem many times at once, see the aggregated answer form in real time, and run control tests that compare each scaling method against a single-sample baseline.

<img src="assets/live-view.jpg" alt="Live view: sixteen samples streaming, an answer tally and the aggregate-correctness curve" width="900" />

<sub>Live view running best-of-N with N = 16 (shown here on the offline mock backend).</sub>

## What you can do

- **Live**: pick a question (or type your own with an expected answer), a method and a budget N, then watch:
  - every sample streaming side by side, each with its extracted answer, ✓/✗ and the verifier (PRM) score;
  - the answer tally re-ranking as samples finish;
  - a step chart of whether the aggregate is right after k samples;
  - beam search as a growing tree (pruned branches fade) and sequential revisions as a chain of attempts.
- **Bench**: run a stratified MATH-500 control test and get:
  - accuracy vs. generation budget for every method;
  - the pass@1 baseline;
  - a compute-optimal line (best method per difficulty level, picked with two-fold cross-validation);
  - accuracy by difficulty level, cost per question, and CSV/JSON export.

  Runs are resumable and locked to their model/PRM/sampling config.

| Method | What it does | Budget N means |
|---|---|---|
| Majority vote | N independent samples; most common answer wins | N samples |
| Best-of-N weighted | N samples; each answer's PRM scores are summed, highest wins | N samples |
| Beam search | Step-by-step; the PRM keeps the best N/4 partial solutions, each expanded 4 ways | N generations per step |
| Sequential revisions | Each attempt sees the last 4 attempts and tries to fix them; PRM-weighted vote over the chain | N attempts |

## Running it

The website runs on your machine; the models run on a Colab GPU and reach the site through a temporary tunnel.

**1. Local app (Python 3.11+)**

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # macOS/Linux: .venv/bin/pip
.venv/Scripts/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. MATH-500 is downloaded to `data/` on first start. The app starts on the **mock backend**: a fake model that lets you try every view offline. Its results are labelled MOCK everywhere.

**2. Colab GPU**

1. Upload `colab/tts_server.ipynb` to Colab and choose a GPU runtime (A100 or L4 recommended; a T4 works but is slow).
2. **Run all.** The first run installs vLLM and downloads the models (5–15 min).
3. Click the backend pill in the site's top-right corner, choose **Colab**, and paste the **URL** and **TOKEN** the last cell printed.

The URL changes every Colab session. If Colab disconnects mid-benchmark, the run pauses; reconnect and press **Resume** to continue where it stopped. To compare against a bigger model, set `POLICY_MODEL` in the notebook (e.g. `Qwen/Qwen3-8B` on an A100), run a **reference-model run** in Bench, and overlay its pass@1 on your main run's chart.

## Differences from the paper

- **Verifier.** The paper trains its own PRM; this uses the off-the-shelf `Qwen/Qwen2.5-Math-PRM-7B` (loaded in 4-bit on GPUs under 40 GB). A solution's score is its last-step score, as in the paper.
- **Revisions.** The paper fine-tunes a revision model and selects with a trained verifier; here revisions are *prompted* and the answer is a PRM-weighted vote over the chain.
- **Difficulty.** Compute-optimal selection uses MATH's own difficulty levels (oracle difficulty) rather than a learned difficulty estimate.
- **Model.** Qwen3 runs in non-thinking mode with steps separated by blank lines, so the PRM can score each step.

## Project layout

```
app/           local FastAPI app: methods, grading, metrics, bench runner, SQLite store
  methods/     sampling.py (majority, best-of-N), beam.py, revisions.py
web/           the website (plain HTML + ES modules + Chart.js)
colab/         gateway.py (served from Colab) and the generated tts_server.ipynb
tests/         pytest suite; runs entirely on the mock backend, no GPU needed
docs/          design and implementation plan
```

Run the tests with `.venv/Scripts/python -m pytest`. After editing `colab/gateway.py`, regenerate the notebook with `python colab/build_notebook.py`.

## Reference

This work is inspired by Google DeepMind's research on inference-time compute (test-time scaling):

- Snell, C., Lee, J., Xu, K., & Kumar, A. (2024). *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*. [arXiv:2408.03314](https://arxiv.org/abs/2408.03314)
