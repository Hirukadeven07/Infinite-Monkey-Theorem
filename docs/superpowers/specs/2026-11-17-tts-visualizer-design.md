# Design: Test-Time Scaling Visualizer (Qwen3 × MATH-500)

## Context

The repo (`Infinite Monkey Theorem`) currently holds only a README and the paper
*Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model
Parameters* (Snell et al., 2024) in `paper/`. The README states the project aim:
study how spending more compute at inference time changes a model's performance.

The user wants to see that for themselves on a Qwen3 small model:
- **Watch test-time scaling happen live**: many responses to one question
  streaming in at once, the aggregated answer forming, verifier scores and search
  beams updating in real time.
- **Run control tests** inside the same UI and get metrics that compare the scaled
  model against a baseline.
- A simple **localhost website**, with the heavy compute in a **Colab Pro notebook**.

Constraints found during recon:
- Local machine: Ryzen 7 5800U laptop, integrated graphics, 16 GB RAM, Python
  3.13, Ollama installed (no Qwen3 pulled), and fastapi, torch and transformers
  installed. It is too slow for N=16–64 sampling, so the GPU work happens in Colab.
- The paper's best methods need a process reward model (PRM) verifier, and a GPU
  makes that feasible.

## Decisions made with the user

| Decision | Choice |
|---|---|
| Where things run | **Local website on localhost + Colab GPU** serving the models through a tunnel URL the user pastes into the site |
| Control test set | **MATH-500** (the paper's benchmark, with difficulty levels 1–5) |
| Methods | **The paper's set**: majority vote, best-of-N weighted by the PRM, beam search with the PRM, sequential revisions |

### Approaches considered (where the model runs)
- **A. Local site + Colab GPU (chosen).** localhost UI. The notebook runs the
  models and exposes them via a Cloudflare quick tunnel. You get a real GPU and
  local control. The trade-off: you paste a new URL each Colab session.
- B. Everything in Colab, opened through Colab's proxy link. Simpler, but it isn't
  localhost and the UI dies with the session.
- C. Everything local on CPU via Ollama. Too slow for meaningful metrics at
  N=16–64.

## Goals
1. **Live view**: pick a MATH-500 question (or type a custom one with an expected
   answer), pick a method and budget N, then watch:
   - N sample cards streaming tokens in parallel, each showing its extracted
     answer, PRM score and a ✓/✗ against ground truth;
   - an answer tally (votes, or PRM-weighted votes) updating as samples finish,
     with the ground-truth answer highlighted;
   - for beam search, a **beam tree** growing step by step, with pruned branches
     greyed out and PRM scores on the nodes;
   - for revisions, a **chain** of attempts left→right with each attempt's answer
     and correctness;
   - running counters: tokens generated, wall time, tokens/s;
   - the baseline (a single sample) shown alongside for contrast.
2. **Bench view (control tests)**: choose a question subset (default: 50
   questions stratified 10 per level, fixed seed), the methods, and the budgets
   (1, 2, 4 … 64). Run it with live progress and an ETA. Results:
   - **accuracy vs generation budget** chart (log₂ x-axis), one line per method,
     plus a dashed **baseline** line;
   - an accuracy table by **difficulty level × method** at the selected budget;
   - a **compute-optimal** line: for each level and budget, the best method
     (using MATH's own levels as oracle difficulty), which mirrors the paper's
     central result;
   - tokens and wall time per method;
   - CSV/JSON export.
3. **Colab notebook** that loads the models, starts the server, and prints the
   tunnel URL and an access token in one "Run all".

## Non-goals
- Training or fine-tuning anything. The paper fine-tuned a revision model and a
  PRM. We use an off-the-shelf PRM and *prompted* revisions, and the UI says so.
- Lookahead search (the paper found it worse per unit of compute), and
  learned difficulty prediction (we use MATH levels as oracle difficulty).
- Multi-user use, auth beyond a token, or deployment beyond localhost.

## Anti-goals (what would make it a failure even if it works)
- A "live" view that is really a replay, or one that hides token cost.
- Metrics that silently mix runs with different model, sampling or prompt
  settings. Every run records its full config, and charts only combine runs with
  matching configs.
- Unreproducible numbers: subsets and subsampling use fixed seeds.

## Architecture

```
Browser (localhost:8000) ──WebSocket──▶ Local FastAPI app (orchestrator)
                                          │  methods, grading, metrics, SQLite
                                          │  one WebSocket, Bearer token
                                          ▼
                           Cloudflare quick tunnel (trycloudflare URL)
                                          ▼
                      Colab gateway (FastAPI :9000)
                       ├─ /v1/*  → proxy to vLLM OpenAI server (:8001), policy model
                       ├─ /prm/score → Qwen2.5-Math-PRM-7B (transformers, in-process)
                       └─ /health → models loaded, GPU, versions
```

- **Tunnel transport (decided after review).** Cloudflare quick tunnels don't
  support SSE and cap in-flight requests at 200. So the local app and the Colab
  gateway talk over **one persistent WebSocket** (`/ws`, token checked when the
  connection opens) that carries all generate and score jobs, each tagged with a
  job id. The gateway streams vLLM's output to the client as WebSocket messages.
  This needs no account, and many requests travel over one connection.
- **The orchestration lives locally.** Majority vote, BoN, the beam search loop,
  revisions, grading and metrics all run in the local app. Colab stays a thin
  model server, the method logic can be tested locally, and the UI owns the event
  stream. Beam search makes one round trip per step, and tunnel latency (about
  100–300 ms) is small next to generation time.
- **Backend abstraction**: the local app talks to any OpenAI-compatible endpoint
  plus a PRM scorer. Two backends:
  - `colab` (real runs);
  - `mock`, a deterministic fake policy and PRM, used for tests and for trying
    the UI offline. Its results are labelled and excluded from Bench charts.
  - Local Ollama was dropped at the user's choice: its quantized weights make
    results incomparable with Colab runs.

## Model and sampling details
- **Policy**: `Qwen/Qwen3-1.7B` by default. It can be switched to 0.6B or 4B with
  the notebook variable `POLICY_MODEL`.
- Qwen3 runs in **non-thinking mode** (`enable_thinking=False`) with the prompt
  "Solve step by step. Separate steps with a blank line. Put the final answer in
  \boxed{}." Steps split on `\n\n`, which matches the Qwen PRM's step format and
  keeps outputs short enough for N=64. *(My call: thinking mode makes outputs
  very long and the steps hard to separate for the PRM. It could become a later
  toggle for majority vote only.)*
- Sampling follows Qwen3's non-thinking defaults: T=0.7, top_p=0.8, top_k=20,
  max_tokens=2048. All of these are recorded in the run config.
- **PRM**: `Qwen/Qwen2.5-Math-PRM-7B`, scored per step with the model card's
  `<extra_0>` separator method. A solution's score is its **last-step** score,
  as in the paper. On an A100 it loads in bf16. On an L4 or T4 the notebook
  loads it in 4-bit (bitsandbytes) automatically, based on detected GPU memory.
- **Baseline**:
  - Primary: the same model, single sample (**pass@1**), estimated as the mean
    single-sample accuracy over the drawn samples.
  - Optional: a **reference-model run**, for example `Qwen3-8B` with one sample
    in a separate notebook session. This tests the paper's headline claim that a
    small model plus test-time compute can match a larger model.

## Methods
| Method | How | Budget N means |
|---|---|---|
| Majority vote | N independent samples; most common normalized answer wins | N samples |
| Best-of-N weighted | N samples; answers' PRM scores are summed per distinct answer, highest total wins (the paper's "best-of-N weighted") | N samples |
| Beam search | Generate one step at a time (stop at `\n\n`); keep N/M beams by PRM score, expand each with M continuations; M=4 by default; max 40 steps | N generations per step |
| Sequential revisions | Chain of N attempts; each prompt includes the **last 4** attempts (as in the paper, to stay inside the 32K context) and asks the model to check them and write a corrected solution; final answer = **PRM-weighted vote over the chain** (the paper used a trained verifier here; we substitute the PRM), with plain majority also reported | N attempts |

- **Efficient curves.** For majority and BoN, draw 64 samples per question once.
  Curves for N ∈ {1, 2, 4 … 64} come from repeated subsampling of those 64 (fixed
  seed, averaged), as the paper does. Beam search and revisions are run once per
  budget, with default budgets {4, 16} for beam (64 as an option) and {4, 8}
  for revisions.
- **Default bench size (decided after review)**: 50 questions, 10 per level, to
  keep a full default bench within a few hours of A100/L4 time. 100 or all 500
  are options.
- **Resumable benches**: each (question, method, budget) result is saved to local
  SQLite as soon as it finishes. If Colab disconnects, the user starts a new
  session, pastes the new URL and token, and clicks Resume, which skips completed
  items.
- **Config-locked results**: a run's config includes policy model, PRM model and
  **PRM precision** (bf16 or 4-bit), GPU name, sampling settings, prompt version
  and vLLM version. Charts only combine results with identical configs.
- **Compute-optimal line**: picks the best method per (level, budget) with
  **two-fold cross-validation** over the bench questions, as the paper does, so
  the line isn't fitted to the questions it reports on. It chooses among the
  four methods (and beam widths if more than one was run).
- **Grading**: extract the last `\boxed{…}`, then check equivalence with Hugging
  Face's `math-verify` library against the MATH-500 `answer` field. math-verify's
  default timeouts rely on Unix `signal` alarms, which Windows doesn't have and
  which don't work outside the main thread. So grading runs with its timeouts
  turned off, inside a small process pool, with a timeout per task.
- **GPU notes**: recommend A100 or L4. A T4 works, but the policy runs in fp16,
  the PRM in 4-bit, and it's slow. The notebook prints the detected tier and the
  precision.

## Data and storage
- MATH-500 comes from the Hugging Face dataset `HuggingFaceH4/MATH-500`
  (`problem`, `answer`, `level`, `subject`). It is downloaded once and cached to
  `data/math500.jsonl` (gitignored).
- Runs are stored in SQLite at `runs/tts.db` (gitignored), with tables for runs
  (full config), questions, samples (text, answer, correct, PRM step scores,
  tokens, latency) and beam nodes. Metrics are computed from stored samples, so
  curves can be recomputed without re-generating.

## UI (plain HTML + vanilla JS modules + Chart.js; served by the local FastAPI app, no build step)

```
┌ Backend ● Colab · Qwen3-1.7B · PRM ✓ (A100) [change]     [ Live ] [ Bench ] ┐
│ Question [MATH-500 #212 · Level 4 · Algebra ▾] [random] [custom…]           │
│ Method [Best-of-N weighted ▾]  N [16 ▾]  [ Run ] [ Stop ]                    │
├─────────────────────────────────────┬───────────────────────────────────────┤
│ ┌#1─────────┐┌#2─────────┐┌#3──────┐│ Answer tally (PRM-weighted)           │
│ │Step 1: … ▍││Step 1: …  ││Step 1:▍││  12  ███████████  ← ground truth ✓    │
│ │PRM ▮▮▮▯ .82││PRM ▮▯▯▯ .31││        ││  15  ████                             │
│ │→ 12  ✓    ││→ 15  ✗    ││        ││   7  █                                │
│ └───────────┘└───────────┘└────────┘│ Aggregate correct after k samples     │
│  … (beam tree / revision chain       │  ▁▁▇▇▇▇▇▇  baseline (1 sample): ✗     │
│      replaces the grid for those)    │ 8,412 tok · 41 s · 205 tok/s          │
└─────────────────────────────────────┴───────────────────────────────────────┘

Bench:  Set MATH-500 · 50 q (10/level) · seed 0   Methods ☑maj ☑BoN ☑beam ☑rev
        Budgets 1 2 4 8 16 32 64   [ Run ]   ███████░░░ 32/50 · ETA 18 min
        [accuracy vs budget chart, dashed pass@1 baseline, compute-optimal line]
        [level × method table]  [export CSV/JSON]
```

## Repo layout (new)
```
app/            local FastAPI app
  main.py         routes, WebSocket, static files
  backends.py     OpenAI-compatible client, PRM client, mock backend
  methods/        majority.py, best_of_n.py, beam.py, revisions.py (emit events)
  grading.py      boxed extraction + math-verify
  data.py         MATH-500 download/cache, stratified subsets
  store.py        SQLite
  metrics.py      subsampled curves, per-level tables, compute-optimal
web/            index.html, live.js, bench.js, styles.css
colab/tts_server.ipynb   installs vllm, loads policy+PRM, gateway, cloudflared, prints URL+token
tests/          pytest; method logic and metrics against the mock backend
requirements.txt, .gitignore (data/, runs/)
```

## Security
- The local app binds to 127.0.0.1 only.
- The Colab gateway requires a random bearer token that the notebook generates and
  prints. The tunnel URL is public, so every gateway route checks the token.
- The token is stored only in the browser's localStorage and the local app's
  memory, never in the repo.

## Calls I made (check these while reading)
- Qwen3 in non-thinking mode; steps split on blank lines.
- One WebSocket over the Cloudflare quick tunnel (no account needed) instead of
  SSE.
- Default bench is 50 questions (10/level); beam budgets {4, 16}, revisions
  {4, 8}.
- Revisions keep the last 4 attempts in context and use a PRM-weighted vote in
  place of the paper's trained verifier.
- Compute-optimal line uses two-fold cross-validation.
- PRM is 4-bit automatically on L4/T4, and results with different precision
  never share a chart.

## Left to the builder
- Exact WebSocket event schema, CSS styling and chart details.
- vLLM `gpu_memory_utilization` split and versions pinned in the notebook.
- Number of subsampling repeats (suggest 100), and bench concurrency (how many
  questions in flight at once).

## Verification
1. `pytest`: grading (LaTeX equivalence cases), vote and BoN aggregation, beam
   bookkeeping and subsampled curves, all against the mock backend.
2. `uvicorn app.main:app` with the **mock** backend. Open localhost, run each
   method in Live (cards stream, tally updates, beam tree grows), and run a
   10-question Bench (chart and table render, export works).
3. Colab: Run all → copy URL and token into the site → `/health` shows both
   models → Live run of BoN N=16 on a Level-3 question → a 20-question Bench at
   budgets {1, 4, 16}. Sanity check: pass@1 for Qwen3-1.7B on MATH-500 lands in
   a plausible range, and majority/BoN accuracy rises with N.
