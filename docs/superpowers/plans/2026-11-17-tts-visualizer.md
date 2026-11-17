# Test-Time Scaling Visualizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a localhost website that streams many Qwen3 answers to a MATH-500 question live, shows majority vote / PRM best-of-N / beam search / revisions forming an answer, and runs benchmark comparisons against a single-sample baseline. Models are served from a Colab GPU notebook.

**Architecture:** A local FastAPI app does all the orchestration: methods, grading, metrics and SQLite. It serves a no-build vanilla-JS frontend and talks to a thin Colab "gateway" over a single authenticated WebSocket through a Cloudflare quick tunnel. The gateway proxies a vLLM OpenAI server (policy) and hosts the Qwen PRM in-process. A deterministic mock backend implements the same interface for tests and offline UI work.

**Tech Stack:** Python 3.13 (local), FastAPI, uvicorn, httpx, websockets, math-verify 0.9.0, pytest + pytest-asyncio, SQLite (stdlib), Chart.js 4.4.1 (CDN), vLLM (Colab), transformers + bitsandbytes (Colab PRM), cloudflared.

**Spec:** `docs/superpowers/specs/2026-11-17-tts-visualizer-design.md`

## Global Constraints

- Commits carry **no** `Co-Authored-By` line (user preference). Never commit `paper/`, `data/`, `runs/`, `.venv/`.
- Local app binds `127.0.0.1` only. Run: `.venv/Scripts/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`.
- Sampling defaults: `temperature=0.7, top_p=0.8, top_k=20, max_tokens=2048`.
- Qwen3 non-thinking prompt; steps split on `"\n\n"`; prompt instruction text (exact): `Solve the following math problem step by step. Separate steps with a blank line. Put the final answer in \boxed{}.`
- PRM solution score = **last-step** score. PRM = `Qwen/Qwen2.5-Math-PRM-7B`; bf16 if GPU total memory ≥ 40 GB, else 4-bit.
- Beam: `M=4`, max 40 steps, budget N must be a multiple of M and ≥ M. Revisions: last 4 attempts in context; final = PRM-weighted vote over chain, plain majority also reported.
- Bench defaults: 10 questions per level (50 total), seed 0, pool size 64 (majority/BoN curves at 1,2,4…64 by subsampling, 100 repeats, seed 0), beam budgets [4,16], revision budgets [4,8]. Compute-optimal uses two-fold cross-validation.
- Results lock to config: policy model, PRM model, PRM precision, vLLM version, sampling, `PROMPT_VERSION`. GPU name is recorded per item but not locked (my call: resume on a different GPU is allowed if precision matches).
- Mock-backend runs are labelled `MOCK` everywhere and never overlaid with real runs.
- Frontend: plain HTML + ES modules, no build step; Chart.js from `https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js`.
- `colab/gateway.py` must import only stdlib + fastapi/httpx at module level (torch/transformers imported lazily) so it is testable locally.

## Review Focus

1. **Sample with no `\boxed{}`** (truncated at max_tokens): answer is `None`, graded wrong, shown as "(no answer)", and never beats a real answer in a vote. Test: Task 3.
2. **Colab disconnects mid-run**: pending jobs raise `BackendError`; Live sends an `error` event; the Bench item isn't marked done, the run goes to `paused`, and Resume continues. Tests: Task 11 and Task 9.
3. **User clicks Stop mid-stream**: the local app sends `cancel` for open jobs, and the gateway aborts the vLLM stream and replies `end`. Tests: Task 10 and Task 11.
4. **Pasted tunnel URL variants** (`https://…/`, `…/ws`, `http://`): normalized to `wss://host/ws`. Test: Task 11.
5. **Resume under a different config** (e.g. 4-bit PRM after bf16): refused with a message naming the differing keys. Test: Task 9.

---

### Task 1: Scaffold, shared schema, prompts, MATH-500 data

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `.gitignore`, `app/__init__.py`, `app/methods/__init__.py`, `colab/__init__.py`, `tests/__init__.py`
- Create: `app/schema.py`, `app/prompts.py`, `app/data.py`
- Test: `tests/test_data_prompts.py`, fixture `tests/fixtures/math500_sample.jsonl` (10 rows, 2 per level, real MATH-500 format)

**Interfaces — Produces:**
- `app/schema.py` (dataclasses):
  - `Question(id: str, problem: str, answer: str, level: int, subject: str)`
  - `Sampling(temperature=0.7, top_p=0.8, top_k=20, max_tokens=2048)` with `to_dict()`
  - `Delta(index: int, text: str)`, `Done(index: int, text: str, tokens: int, finished: bool)`. `finished=False` means it stopped at a stop string (a step boundary); `True` means EOS or the length limit.
  - `SampleRecord(idx: int, text: str, answer: str | None, correct: bool, prm_steps: list[float] | None, tokens: int, seconds: float)` with property `prm -> float | None` (the last step).
  - `MethodResult(method: str, budget: int, final_answer: str | None, correct: bool, majority_answer: str | None, tokens: int, seconds: float, samples: list[SampleRecord], detail: dict)`
  - `class BackendError(Exception)`
- `app/prompts.py`: `PROMPT_VERSION = "v1"`, `INSTRUCTION` (exact text above), `solve_prompt(problem: str) -> str`, `revision_prompt(problem: str, previous: list[str]) -> str` (uses only `previous[-4:]`), `split_steps(text: str) -> list[str]`.
  - Raw Qwen3 non-thinking template: `"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"`. For solve, content is `f"{INSTRUCTION}\n\n{problem}"`.
  - Revision content: `INSTRUCTION`, the problem, then each previous attempt as `"Previous attempt {i}:\n{text}"`, then `"Check the previous attempts for mistakes and write a corrected, complete solution."`
- `app/data.py`: `MATH500_URL = "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl"`, `load_math500(path: Path = Path("data/math500.jsonl")) -> list[Question]` (downloads with httpx if missing; `id = unique_id`), `stratified_subset(questions, per_level: int, seed: int) -> list[Question]` (sorted by level, deterministic).
- `requirements.txt`: `fastapi uvicorn[standard] httpx websockets math-verify[antlr4_13_2]==0.9.0 pytest pytest-asyncio`. `pytest.ini`: `asyncio_mode = auto`, `testpaths = tests`. `.gitignore`: `.venv/ data/ runs/ __pycache__/ paper/`

- [ ] **Step 1: Create venv and install** — `python -m venv .venv && .venv/Scripts/pip install -r requirements.txt`. Expected: installs cleanly on 3.13.
- [ ] **Step 2: Write failing tests**

```python
def test_solve_prompt_is_qwen3_non_thinking():
    p = solve_prompt("What is 1+1?")
    assert p.startswith("<|im_start|>user\nSolve the following math problem step by step.")
    assert p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "What is 1+1?" in p

def test_revision_prompt_keeps_last_four():
    p = revision_prompt("Q", [f"att{i}" for i in range(6)])
    assert "att0" not in p and "att1" not in p and "att2" in p and "att5" in p

def test_split_steps_drops_empty():
    assert split_steps("a\n\nb\n\n\n\nc\n\n") == ["a", "b", "c"]

def test_load_math500_from_fixture(tmp_path):
    qs = load_math500(FIXTURE)
    assert len(qs) == 10 and {q.level for q in qs} == {1, 2, 3, 4, 5}
    assert all(q.id and q.answer for q in qs)

def test_stratified_subset_deterministic():
    qs = load_math500(FIXTURE)
    a, b = stratified_subset(qs, 1, seed=0), stratified_subset(qs, 1, seed=0)
    assert [q.id for q in a] == [q.id for q in b]
    assert sorted(q.level for q in a) == [1, 2, 3, 4, 5]
```

- [ ] **Step 3: Run** `.venv/Scripts/python -m pytest tests/test_data_prompts.py -v`. Expected: FAIL (import errors).
- [ ] **Step 4: Implement** schema, prompts and data as specified.
- [ ] **Step 5: Run tests.** Expected: PASS.
- [ ] **Step 6: Commit** `git add requirements.txt pytest.ini .gitignore app colab/__init__.py tests && git commit -m "Scaffold app with schema, prompts and MATH-500 loader"`

---

### Task 2: Grading

**Files:** Create `app/grading.py`; Test `tests/test_grading.py`

**Interfaces — Produces:**
- `extract_boxed(text: str) -> str | None`: content of the **last** `\boxed{…}` or `\fbox{…}`, with balanced braces (nested braces handled).
- `is_correct(pred: str | None, gold: str) -> bool`: `None` gives False. Uses `math_verify.parse(f"\\boxed{{{x}}}", parsing_timeout=None)` for both sides and `verify(gold_parsed, pred_parsed, timeout_seconds=None)`. Any exception gives False.
- `class Grader`: `__init__(workers: int = 2, timeout: float = 10.0)`; `async def grade(pred: str | None, gold: str) -> bool` runs `is_correct` in a `ProcessPoolExecutor` (top-level function, so Windows spawn works), caches by `(pred, gold)`, and returns False on timeout; `close()`.

- [ ] **Step 1: Write failing tests**

```python
def test_extract_last_boxed_nested():
    assert extract_boxed(r"x \boxed{1} then \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
def test_extract_none():
    assert extract_boxed("no answer here") is None
@pytest.mark.parametrize("pred,gold", [(r"\frac12", r"\frac{1}{2}"), ("0.5", r"\frac{1}{2}"),
    (r"\dfrac{3}{4}", r"\frac{3}{4}"), ("(1,2)", r"(1, 2)"), (r"2\sqrt{2}", r"\sqrt{8}")])
def test_equivalent(pred, gold): assert is_correct(pred, gold)
def test_not_equivalent(): assert not is_correct("3", "4")
def test_none_and_garbage_are_wrong():
    assert not is_correct(None, "4")
    assert not is_correct("4", r"\frac{{{")      # unparseable gold must not raise
async def test_grader_pool():
    g = Grader(workers=1)
    assert await g.grade("0.5", r"\frac{1}{2}") is True
    assert await g.grade(None, "1") is False
    g.close()
```

- [ ] **Step 2: Run** `pytest tests/test_grading.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** If one parametrized equivalence case fails because of math-verify semantics, check it in a REPL. Drop only cases math-verify genuinely treats as unequal, and note them in the commit message.
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "Add answer extraction and math-verify grading"`

---

### Task 3: Answer aggregation

**Files:** Create `app/aggregate.py`; Test `tests/test_aggregate.py`

**Interfaces — Produces:**
- `answer_key(ans: str | None) -> str | None`: canonical string for grouping. Strip whitespace, `\left`, `\right`, `\!`, `\,`, trailing `.`, `^\circ` and `^{\circ}`; unwrap `\text{…}`; `\dfrac` and `\tfrac` become `\frac`.
- `majority(answers: list[str | None]) -> str | None`: the most common key. Ties go to the earliest first occurrence. `None` is ignored unless all are `None`. Returns the **original** text of the first member of the winning group.
- `weighted_vote(answers: list[str | None], scores: list[float]) -> str | None`: the same, but each group's weight is the sum of its scores.
- `tally(answers, scores: list[float] | None, correct: list[bool]) -> list[dict]`: one entry per key, `{"answer", "votes", "weight", "correct"}`, sorted by weight (or votes) descending. A `None` answer gets the entry `"answer": "(no answer)"`.

- [ ] **Step 1: Write failing tests**

```python
def test_majority_groups_equivalent_forms():
    assert answer_key(r"\dfrac{1}{2}") == answer_key(r"\frac{1}{2}")
    assert majority([r"\frac{1}{2}", "3", r"\dfrac{1}{2}"]) == r"\frac{1}{2}"
def test_majority_tie_earliest_wins(): assert majority(["3", "5"]) == "3"
def test_none_never_beats_real_answer():                # Review Focus 1
    assert majority([None, None, "3"]) == "3"
    assert weighted_vote([None, None, "3"], [0.9, 0.9, 0.1]) == "3"
    assert majority([None, None]) is None
def test_weighted_vote_sums_scores():
    assert weighted_vote(["3", "3", "5"], [0.2, 0.2, 0.9]) == "5"
def test_tally_shape():
    t = tally(["3", None, "3"], None, [True, False, True])
    assert t[0] == {"answer": "3", "votes": 2, "weight": 2.0, "correct": True}
    assert t[1]["answer"] == "(no answer)"
```

- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement.** **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "Add answer normalization, voting and tallies"`

---

### Task 4: Mock backend

**Files:** Create `app/backends.py` (the `Backend` protocol), `app/mock.py`; Test `tests/test_mock.py`

**Interfaces — Produces:**
- `app/backends.py`:
```python
class Backend(Protocol):
    kind: str                       # "colab" | "mock"
    info: dict                      # policy_model, prm_model, prm_precision, gpu, vllm_version, mock: bool
    def generate(self, prompt: str, n: int, sampling: Sampling,
                 stop: list[str] | None = None) -> AsyncIterator[Delta | Done]: ...
    async def score(self, problem: str, solutions: list[list[str]]) -> list[list[float]]: ...
    async def close(self) -> None: ...
```
  The `generate` contract: it interleaves `Delta`s and exactly one `Done` per index (0..n-1). Closing the iterator early (`aclose`) cancels the job.
- `app/mock.py`: `MockBackend(questions: list[Question], seed: int = 0, delay: float = 0.0, steps: int = 4)`, `kind="mock"`, `info={"policy_model": "mock", "prm_model": "mock", "prm_precision": "mock", "gpu": "none", "vllm_version": "mock", "mock": True}`.
  - **Behaviour.** Find the question whose `problem` occurs in the prompt. The assistant prefix is the text after the last `"</think>\n\n"`. Each sample `index` gets an RNG seeded by `hash((seed, prompt, index))` via `hashlib`, not Python's `hash`.
  - `p_correct` by level: `{1: .8, 2: .65, 3: .5, 4: .35, 5: .2}`. A wrong answer is drawn from `[gold+"1", "0", "-1"]`.
  - The full solution is `steps` lines `"Step k: mock reasoning."` joined by `"\n\n"`, with the last step `"So the answer is \\boxed{X}."`.
  - With `stop=["\n\n"]`: emit only the next step after the steps already in the prefix, with `finished = (that step is the last one)`.
  - Stream word by word as `Delta`s, sleeping `delay` per word. `tokens` = word count.
  - `score`: per step, a deterministic hash-noise value. Intermediate steps get 0.6±0.1. The final step gets 0.75±0.15 if its boxed answer is correct, else 0.4±0.15. All values are clipped to [0,1].

- [ ] **Step 1: Write failing tests**

```python
async def collect(it):
    out = [e async for e in it]; return out
async def test_generate_n_streams_and_one_done_each():
    mb = MockBackend(QS); ev = await collect(mb.generate(solve_prompt(QS[0].problem), 4, Sampling()))
    dones = [e for e in ev if isinstance(e, Done)]
    assert sorted(d.index for d in dones) == [0, 1, 2, 3] and all(d.finished for d in dones)
    assert all(extract_boxed(d.text) for d in dones)
async def test_deterministic():
    a = await collect(MockBackend(QS).generate(solve_prompt(QS[0].problem), 3, Sampling()))
    b = await collect(MockBackend(QS).generate(solve_prompt(QS[0].problem), 3, Sampling()))
    assert [e for e in a if isinstance(e, Done)] == [e for e in b if isinstance(e, Done)]
async def test_stop_yields_one_step_until_final():
    mb = MockBackend(QS, steps=3); prefix = solve_prompt(QS[0].problem); finished = False; n = 0
    while not finished:
        d = [e for e in await collect(mb.generate(prefix, 1, Sampling(), stop=["\n\n"])) if isinstance(e, Done)][0]
        prefix += d.text + "\n\n"; finished = d.finished; n += 1
    assert n == 3
async def test_score_shape_and_range():
    s = await MockBackend(QS).score("p", [["a", "b"], ["c"]])
    assert [len(x) for x in s] == [2, 1] and all(0 <= v <= 1 for x in s for v in x)
```

- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement.** **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "Add backend protocol and deterministic mock backend"`

---

### Task 5: Parallel sampling method (majority vote and best-of-N weighted)

**Files:** Create `app/methods/context.py`, `app/methods/sampling.py`; Test `tests/test_sampling.py`

**Interfaces:**
- Consumes: `Backend`, `Grader`, `aggregate.*`, `prompts.*`, schema.
- Produces:
  - `RunContext(backend: Backend, grader: Grader, sampling: Sampling, emit: Callable[[dict], Awaitable[None]] = noop)` (dataclass, in `context.py`).
  - `async def run_sampling(ctx, q: Question, n: int, method: Literal["majority", "bon"]) -> MethodResult`.
    - Stream all `n` samples. When a sample is done: extract the answer, grade it, and PRM-score it (`ctx.backend.score(q.problem, [split_steps(text)])`).
    - `final_answer` = `majority` or `weighted_vote` (by last-step PRM) over **all** samples; `majority_answer` is always the majority.
    - Baseline = sample index 0's correctness, stored as `detail["baseline_correct"]`.
- **Event schema** (shared by every method; the frontend in Tasks 13–15 depends on it):
  - `{"type": "run_start", "method", "budget", "question": {id, problem, answer, level, subject}}`
  - `{"type": "sample_delta", "sample": i, "text": str}`
  - `{"type": "sample_done", "sample": i, "answer": str|null, "correct": bool, "prm": float|null, "tokens": int}`
  - `{"type": "tally", "entries": tally(...)}`, sent after every `sample_done`, weighted for bon.
  - `{"type": "agg_curve", "points": [{"k": int, "correct": bool}]}`: whether the aggregate over the first k *completed* samples is correct.
  - `{"type": "stats", "tokens": int, "seconds": float}`
  - `{"type": "baseline", "answer", "correct"}`
  - `{"type": "run_end", "final_answer", "correct", "majority_answer", "tokens", "seconds"}`
  - `{"type": "error", "message"}`

- [ ] **Step 1: Write failing tests** (mock backend, `delay=0`; `events` collected by a list-appending `emit`)

```python
async def test_bon_runs_and_emits():
    res = await run_sampling(ctx, QS[0], 8, "bon")
    assert len(res.samples) == 8 and all(s.prm is not None for s in res.samples)
    types = [e["type"] for e in events]
    assert types[0] == "run_start" and types[-1] == "run_end"
    assert types.count("sample_done") == 8 and types.count("tally") == 8
    assert events[-1]["final_answer"] == res.final_answer
async def test_majority_final_matches_aggregate():
    res = await run_sampling(ctx, QS[0], 8, "majority")
    assert res.final_answer == majority([s.answer for s in res.samples])
async def test_tokens_counted():
    res = await run_sampling(ctx, QS[0], 4, "bon")
    assert res.tokens == sum(s.tokens for s in res.samples) > 0
```

- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement.** **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "Add parallel sampling method with majority and PRM-weighted voting"`

---

### Task 6: Beam search

**Files:** Create `app/methods/beam.py`; Test `tests/test_beam.py`

**Interfaces — Produces:** `async def run_beam(ctx, q, n: int, m: int = 4, max_steps: int = 40) -> MethodResult`. It raises `ValueError` unless `n % m == 0 and n >= m`.
- Additional events: `{"type": "beam_step", "step": t, "nodes": [{"id": str, "parent": str|null, "text": str, "score": float, "kept": bool, "complete": bool}]}`. It also emits `sample_done` for each completed beam, `tally`, `baseline`, `run_end`.
- `detail["nodes"]` holds all nodes; it is stored in `items_done.detail_json` (my call: no separate beam-node table).

**Algorithm** (not determined by the tests, so it's spelled out):
```
prefix = solve_prompt(problem); active = [Node(id="root", text="")]; completed = []
for t in range(max_steps):
    if not active: break
    expand: for each active beam b, generate(prefix + b.text, n = n if t == 0 else m, stop=["\n\n"])
            (step 0 samples n from root; later steps sample m per kept beam; n/m beams kept)
    child.text = b.text + d.text + ("" if d.finished else "\n\n")
    child.complete = d.finished or extract_boxed(d.text) is not None
    score children in one PRM call (split_steps(child.text)); child.score = last step
    completed += complete children
    active = top n//m incomplete children by score (mark kept)
    emit beam_step; stop early if len(completed) >= n
if not completed: completed = active
final_answer = weighted_vote over completed answers by score; majority_answer = majority(...)
```
Also launch **one plain sample** concurrently (`generate(prefix, 1)`) for the baseline event.

- [ ] **Step 1: Write failing tests**

```python
async def test_beam_completes_and_emits_steps():
    res = await run_beam(ctx, QS[0], 8, m=4)
    steps = [e for e in events if e["type"] == "beam_step"]
    assert steps and all(sum(nd["kept"] for nd in s["nodes"]) <= 2 for s in steps)
    assert res.final_answer is not None and res.samples
    assert any(e["type"] == "baseline" for e in events)
async def test_beam_budget_validation():
    with pytest.raises(ValueError): await run_beam(ctx, QS[0], 6, m=4)
    with pytest.raises(ValueError): await run_beam(ctx, QS[0], 2, m=4)
async def test_beam_respects_max_steps():
    ctx.backend = MockBackend(QS, steps=50)
    res = await run_beam(ctx, QS[0], 4, m=4, max_steps=3)
    assert len([e for e in events if e["type"] == "beam_step"]) == 3 and res.samples  # falls back to active beams
```

- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement.** **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "Add PRM-guided beam search"`

---

### Task 7: Sequential revisions

**Files:** Create `app/methods/revisions.py`; Test `tests/test_revisions.py`

**Interfaces — Produces:** `async def run_revisions(ctx, q, n: int) -> MethodResult`.
- Attempt 0 uses `solve_prompt`; attempt i uses `revision_prompt(problem, texts[:i])`. Attempts run one after another and stream as `sample_delta`/`sample_done` with `sample = i`.
- After each attempt, emit `tally` (PRM-weighted) and `agg_curve`.
- `final_answer` = `weighted_vote` over the chain, `majority_answer` = `majority`. Baseline = attempt 0.

- [ ] **Step 1: Write failing tests**

```python
async def test_revisions_sequential_chain():
    res = await run_revisions(ctx, QS[0], 4)
    assert [s.idx for s in res.samples] == [0, 1, 2, 3]
    starts = [e["sample"] for e in events if e["type"] == "sample_delta"]
    assert starts == sorted(starts)                       # strictly sequential
    assert res.final_answer == weighted_vote([s.answer for s in res.samples], [s.prm for s in res.samples])
async def test_revision_prompt_receives_previous(monkeypatch):
    seen = []; orig = ctx.backend.generate
    def spy(prompt, *a, **k): seen.append(prompt); return orig(prompt, *a, **k)
    monkeypatch.setattr(ctx.backend, "generate", spy)
    await run_revisions(ctx, QS[0], 3)
    assert "Previous attempt" not in seen[0] and "Previous attempt" in seen[2]
```

- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement.** **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "Add sequential revisions method"`

---

### Task 8: Metrics

**Files:** Create `app/metrics.py`; Test `tests/test_metrics.py`

**Interfaces — Produces:**
- `ResultRow(question_id: str, method: str, budget: int, accuracy: float, tokens: int, seconds: float)` (dataclass, defined here; `store` imports it).
- `subsample_curve(samples: list[SampleRecord], budgets: list[int], method: Literal["majority", "bon"], repeats: int = 100, seed: int = 0) -> dict[int, float]`. For each k, draw `repeats` subsets of size k without replacement (only one subset when k == len). Aggregate each subset with `majority` or `weighted_vote(prm)`. The aggregate is correct iff some sample in the subset with the same `answer_key` is correct. Returns the mean.
- `pass_at_1(samples) -> float`: the mean of `correct`.
- `pool_rows(question_id, samples, budgets, repeats, seed) -> list[ResultRow]`: majority and bon rows at each budget, plus a `("pass@1", 1)` row. Tokens are estimated as mean sample tokens × k.
- `curves(rows) -> dict[str, dict[int, float]]`: mean accuracy per method and budget.
- `level_table(rows, levels: dict[str, int]) -> list[dict]`: `{"level", "method", "budget", "accuracy", "n"}`.
- `compute_optimal(rows, levels, seed: int = 0) -> dict[int, float]`. Assign questions to 2 folds, stratified by level (shuffle each level with the seed, then alternate). For each budget B and fold f: on the *other* fold, pick per level the method (excluding `pass@1`) with the highest mean accuracy at B. Score fold-f questions with that method's row. Return the mean over all questions with a row at B.

- [ ] **Step 1: Write failing tests**

```python
def S(ans, ok, prm=0.5): return SampleRecord(0, "", ans, ok, [prm], 10, 0.1)
def test_subsample_full_pool_exact():
    pool = [S("3", True), S("3", True), S("5", False)]
    assert subsample_curve(pool, [3], "majority")[3] == 1.0
def test_subsample_k1_equals_pass1():
    pool = [S("3", True), S("5", False), S("5", False), S("7", False)]
    assert abs(subsample_curve(pool, [1], "majority", repeats=4000)[1] - 0.25) < 0.03
def test_bon_prefers_high_prm():
    pool = [S("3", True, .9), S("5", False, .2), S("5", False, .2)]
    assert subsample_curve(pool, [3], "bon")[3] == 1.0 and subsample_curve(pool, [3], "majority")[3] == 0.0
def test_compute_optimal_picks_per_level_best():
    levels = {f"a{i}": 1 for i in range(4)} | {f"b{i}": 2 for i in range(4)}
    rows = [ResultRow(q, "majority", 4, 1.0 if q[0] == "a" else 0.0, 0, 0) for q in levels] + \
           [ResultRow(q, "bon", 4, 0.0 if q[0] == "a" else 1.0, 0, 0) for q in levels]
    assert compute_optimal(rows, levels)[4] == 1.0
    assert curves(rows)["majority"][4] == 0.5
def test_compute_optimal_uses_held_out_fold():
    levels = {f"q{i}": 1 for i in range(4)}
    # each method is perfect on exactly the questions the other fold would pick it for -> CV must not reach 1.0
    folds = fold_assignment(levels, seed=0)            # expose helper: dict[qid, 0|1]
    rows = [ResultRow(q, "majority", 4, float(folds[q] == 0), 0, 0) for q in levels] + \
           [ResultRow(q, "bon", 4, float(folds[q] == 1), 0, 0) for q in levels]
    assert compute_optimal(rows, levels)[4] < 1.0
```
(`fold_assignment(levels, seed) -> dict[str, int]` is also public.)

- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement.** **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "Add subsampled scaling curves, level tables and cross-validated compute-optimal"`

---

### Task 9: Store and bench runner

**Files:** Create `app/store.py`, `app/bench.py`; Test `tests/test_store.py`, `tests/test_bench.py`

**Interfaces — Produces:**
- `app/store.py`: `Store(path: Path = Path("runs/tts.db"))` (it creates the directory and tables, and uses one `sqlite3` connection with `check_same_thread=False` behind a lock).
  - Tables:
    - `runs(id, created, kind, status, config_json, lock_json)`
    - `samples(run_id, question_id, method, budget, idx, text, answer, correct, prm_json, tokens, seconds)`
    - `results(run_id, question_id, method, budget, accuracy, tokens, seconds, PK(run_id, question_id, method, budget))`
    - `items_done(run_id, question_id, item, budget, gpu, detail_json)`
  - Methods:
    - `create_run(kind: str, config: dict, lock: dict) -> int`, `get_run(id) -> dict`, `list_runs(kind) -> list[dict]`, `set_status(id, status)`
    - `add_samples(run_id, qid, method, budget, samples)`, `save_rows(run_id, rows: list[ResultRow])`, `rows(run_id) -> list[ResultRow]`
    - `mark_done(run_id, qid, item, budget, gpu, detail)`, `done_items(run_id) -> set[tuple[str, str, int]]`
- `app/bench.py`:
  - `BenchConfig(per_level=10, seed=0, pool_size=64, beam_budgets=[4, 16], revision_budgets=[4, 8], methods=["pool", "beam", "revisions"], reference=False, concurrency=4, repeats=100)` (a pydantic model, so FastAPI can parse it).
  - `lock_for(backend_info, sampling) -> dict`: keys `policy_model, prm_model, prm_precision, vllm_version, sampling, prompt_version`. For `reference=True` runs, `policy_model` is still recorded and only the pass@1 baseline is used.
  - `class ConfigMismatch(Exception)` carries `.keys: list[str]`.
  - `class BenchRunner(store, questions: list[Question], grader)`:
    - `start(cfg, backend) -> int` creates the run and launches an asyncio task.
    - `resume(run_id, backend)` raises `ConfigMismatch` (Review Focus 5).
    - `stop(run_id)`, and `progress(run_id) -> {"status", "done", "total", "eta_seconds"}`.
  - Items per question: `("pool", pool_size)` runs `run_sampling(n=pool_size, "bon")` and saves samples plus `pool_rows(budgets=powers of two ≤ pool_size)`; `("beam", b)`; `("revisions", b)`.
  - A reference run only does `("pool", pool_size)` with pool_size from the config (UI default 8), and only its pass@1 is plotted.
  - Items run with `asyncio.Semaphore(concurrency)`. A `BackendError` stops the run with status `"paused"`, and the item is not marked done (Review Focus 2). A finished run gets status `"done"`.

- [ ] **Step 1: Write failing tests** (MockBackend, tmp_path store, 1 question per level, pool_size 4, beam [4], revisions [2])

```python
async def test_bench_completes_and_rows_present(): ...
    # status "done"; done_items has 5 * 3 entries; curves(rows) has majority, bon, pass@1, beam, revisions
async def test_resume_skips_done_items(): ...
    # pre-mark one item done; spy on run_sampling count -> one fewer call
async def test_resume_refuses_config_mismatch():
    run = await finished_run(); other = MockBackend(QS); other.info = {**other.info, "prm_precision": "4bit"}
    with pytest.raises(ConfigMismatch) as e: await runner.resume(run, other)
    assert "prm_precision" in e.value.keys
async def test_backend_error_pauses_run():
    flaky = FlakyBackend(MockBackend(QS), fail_after=3)   # test helper: raises BackendError on 4th generate()
    rid = await runner.start(cfg, flaky); await runner.wait(rid)
    assert store.get_run(rid)["status"] == "paused" and len(store.done_items(rid)) < 15
```
(`BenchRunner.wait(run_id)` awaits the task, which tests need.)

- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement.** **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "Add SQLite store and resumable, config-locked bench runner"`

---

### Task 10: Colab gateway

**Files:** Create `colab/gateway.py`; Test `tests/test_gateway.py`

**Interfaces — Produces** (wire protocol; Task 11's client depends on it exactly):
- Client→server, all JSON over `/ws`:
  - first message `{"op": "auth", "token"}`. A bad token closes the socket with code 4401.
  - `{"op": "generate", "job": str, "prompt", "n", "stop": list|null, "max_tokens", "temperature", "top_p", "top_k"}`
  - `{"op": "score", "job", "problem", "solutions": [[step, …], …]}`
  - `{"op": "cancel", "job"}`
- Server→client:
  - `{"type": "ready", "info": {...}}`
  - `{"job", "type": "delta", "index", "text"}`
  - `{"job", "type": "done", "index", "text", "tokens", "finished"}`
  - `{"job", "type": "end"}` after all dones, or after a cancel
  - `{"job", "type": "scores", "scores": [[float]]}`
  - `{"job", "type": "error", "message"}`
- `GET /health` with header `Authorization: Bearer <token>` returns `info`; it returns 401 otherwise.
- Python:
  - `PolicyClient` protocol: `stream(prompt, n, stop, max_tokens, temperature, top_p, top_k) -> AsyncIterator[dict]` yielding dicts of the delta/done shape above (without `job`).
  - `PrmScorer` protocol: `score(problem, solutions) -> list[list[float]]` (sync; the gateway calls it via `asyncio.to_thread` under a lock).
  - `create_app(policy, prm, token: str, info: dict) -> FastAPI`. Each generate and score op runs as its own task keyed by `job`; `cancel` cancels it and sends `end`.
  - `VllmPolicy(base_url: str, model: str, tokenizer)` posts to `{base_url}/v1/completions` with `stream: true`, the given params, and `n`. It parses SSE `data:` lines and groups by `choice.index`.
    - `finished = not (finish_reason == "stop" and isinstance(choice.get("stop_reason"), str))`
    - `tokens = len(tokenizer(text).input_ids)`
  - `QwenPrm(model_name: str, quant: Literal["bf16", "4bit"])`: lazy imports; `AutoModel.from_pretrained(..., trust_remote_code=True)`, with a `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)` for 4bit. Scoring follows the model card exactly:
```python
msgs = [{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"}]
ids = tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False), return_tensors="pt")
logits = model(input_ids=ids.to(model.device))[0]           # [1, T, 2]
mask = ids.to(model.device) == tok.encode("<extra_0>")[0]
probs = torch.softmax(logits, -1)[0][mask[0]][:, 1]        # per-step P(correct)
```

- [ ] **Step 1: Write failing tests** (FastAPI `TestClient.websocket_connect`, `FakePolicy` yielding 2 deltas + done per index, `FakePrm` returning 0.5s)

```python
def test_bad_token_rejected(): ...          # send auth with wrong token -> WebSocketDisconnect code 4401
def test_generate_streams_then_end(): ...   # n=2: deltas for both indices, 2 dones, then {"type":"end"}
def test_score_roundtrip(): ...             # scores shape matches solutions
def test_concurrent_jobs_multiplexed(): ... # two generate ops interleaved, each gets its own end
def test_cancel_sends_end_and_stops():      # Review Focus 3
    ...  # FakePolicy(slow=True); send generate then cancel -> receive end; FakePolicy.cancelled is True
def test_health_requires_bearer(): ...      # 401 without header, 200 with
```

- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement** (the module imports only stdlib, fastapi, httpx). **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "Add Colab gateway with authenticated multiplexed WebSocket protocol"`

---

### Task 11: Colab backend client

**Files:** Modify `app/backends.py`; Test `tests/test_colab_backend.py`

**Interfaces — Produces:**
- `ws_url(pasted: str) -> str`: strips whitespace and a trailing `/` or `/ws`, maps `https→wss` and `http→ws`, and adds a scheme if missing (wss), then appends `/ws`.
- `class ColabBackend(url: str, token: str)`, `kind="colab"`:
  - `async connect() -> dict` sends auth and waits for `ready`; it raises `BackendError("bad token")` on 4401 and `BackendError(...)` on any connect failure.
  - It satisfies `Backend`. One reader task routes messages to per-job `asyncio.Queue`s.
  - `generate` sends `cancel` when its iterator is closed early.
  - When the socket drops, every pending queue gets a `BackendError`, which the iterator raises.
  - `connected: bool` property.

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.parametrize("raw", ["https://a.trycloudflare.com", "https://a.trycloudflare.com/",
                                 "https://a.trycloudflare.com/ws", " a.trycloudflare.com "])
def test_ws_url_normalizes(raw): assert ws_url(raw) == "wss://a.trycloudflare.com/ws"   # Review Focus 4
def test_ws_url_http(): assert ws_url("http://127.0.0.1:9000") == "ws://127.0.0.1:9000/ws"
async def test_generate_and_score_against_live_gateway(gateway_server): ...
    # gateway_server fixture: uvicorn.Server(create_app(FakePolicy(), FakePrm(), "t", {...})) on a free port in a thread
async def test_bad_token(gateway_server): ...           # BackendError
async def test_disconnect_raises(gateway_server):       # Review Focus 2
    ...  # start slow generate, call gateway_server.should_exit = True -> iterator raises BackendError
async def test_early_close_sends_cancel(gateway_server): ...  # aclose after first delta -> FakePolicy.cancelled
```

- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement.** **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "Add Colab WebSocket backend client"`

---

### Task 12: Local FastAPI app

**Files:** Create `app/main.py`; Test `tests/test_main.py`

**Interfaces — Produces:**
- App state: the current `Backend` (default `MockBackend(questions, delay=0.02)`), `Store`, `Grader`, `BenchRunner`, and questions from `load_math500()`. A `TTS_DATA` env var overrides the data path; tests point it at the fixture.
- Routes:
  - `GET /` serves `web/index.html`; `/static` serves `web/`.
  - `GET /api/backend` returns `{"kind", "connected", "info"}`.
  - `POST /api/backend` takes `{"kind": "colab", "url", "token"} | {"kind": "mock"}`. It connects, swaps the backend, and returns the same shape. On `BackendError` it returns 502 with `{"detail": message}`.
  - `GET /api/questions?level=` returns `[{id, level, subject, problem, answer}]`.
  - `WS /ws/live`. The client sends `{"op": "run", "question_id"? , "custom"?: {"problem", "answer"}, "method": "majority"|"bon"|"beam"|"revisions", "budget": int}` or `{"op": "stop"}`.
    - The server streams the Task 5 events. It keeps one run per socket; `stop` cancels the method task, which closes generate iterators and so sends `cancel`.
    - A `BackendError` or `ValueError` sends an `error` event.
    - Every live run is saved as kind `"live"`.
  - `POST /api/bench` (body `BenchConfig`) returns `{"run_id"}`. `POST /api/bench/{id}/resume` returns 409 with `{"detail", "keys"}` on `ConfigMismatch`. `POST /api/bench/{id}/stop`. `GET /api/runs?kind=bench` lists runs with config and lock.
  - `GET /api/bench/{id}/metrics` returns `{"progress", "curves", "baseline": pass@1, "by_level": level_table, "compute_optimal", "cost": {method: {"tokens": mean per question, "seconds": mean per question}}, "mock": bool, "lock"}`.
  - `GET /api/bench/{id}/export.csv` has columns `run_id,question_id,level,method,budget,accuracy,tokens,seconds`. `GET /api/bench/{id}/export.json` returns rows plus config.

- [ ] **Step 1: Write failing tests** (TestClient; mock backend; fixture data)

```python
def test_live_ws_bon_runs(client):
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"op": "run", "question_id": QID, "method": "bon", "budget": 4})
        types = collect_until(ws, "run_end")
        assert "sample_delta" in types and "tally" in types
def test_live_custom_question(client): ...           # custom problem not in dataset -> mock treats as level 3; run_end arrives
def test_live_bad_beam_budget_sends_error(client): ...  # beam budget 6 -> {"type":"error"}
def test_backend_bad_url_502(client): ...             # POST colab url to closed port -> 502
def test_bench_metrics_and_export(client): ...        # small BenchConfig; poll metrics until status done; csv header exact
```
Mock custom-question support: `MockBackend` treats an unknown problem as level 3 with gold `"42"`. Add this to `app/mock.py` along with a test in `tests/test_mock.py`.

- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement.** **Step 4: Run** `pytest -v` (the full suite). Expected: all PASS.
- [ ] **Step 5: Commit** `git commit -m "Add local FastAPI app with live WebSocket and bench API"`

---

### Task 13: Frontend shell + Live view (sampling methods)

**Files:** Create `web/index.html`, `web/styles.css`, `web/app.js`, `web/live.js`

**Interfaces:** Consumes the Task 12 routes and the Task 5 event schema.
- `app.js`:
  - Tabs `Live | Bench`.
  - Backend bar: a status dot, `kind · policy_model · PRM precision (gpu)`, a `MOCK` badge when `info.mock`, and a [change] dialog with URL + token fields → `POST /api/backend`.
  - The URL and token are remembered in `localStorage` inside try/catch.
- `live.js` controls:
  - question picker grouped by level, with [random] and [custom…] (problem + expected answer)
  - method select (Majority vote / Best-of-N weighted / Beam search / Sequential revisions, with a note under revisions: "prompted revisions + PRM vote; the paper fine-tuned a revision model")
  - budget select (1, 2, 4, 8, 16, 32, 64)
  - Run / Stop
- `live.js` panels:
  - The **sample grid** has one card per sample. Each card shows streaming text with a caret, then a PRM bar + value, then `→ answer ✓/✗`. The answer is rendered as plain text; no KaTeX (my call: YAGNI).
  - The **tally** is horizontal bars with the ground-truth row marked.
  - The **aggregate curve** is a Chart.js step line of `agg_curve` (correct=1/incorrect=0 vs k).
  - The **baseline** line reads "baseline (1 sample): ✓/✗".
  - **Stats** show `tokens · seconds · tok/s`.
  - Errors appear in a banner.
- Token deltas are appended through `requestAnimationFrame` batching so N=64 streams stay smooth.

- [ ] **Step 1: Implement** the files.
- [ ] **Step 2: Verify in the browser.** Start the app (Global Constraints command) with the mock backend and open `http://127.0.0.1:8000`. Run BoN N=16 on a Level-3 question.
  - Expected: 16 cards stream at once, the tally reorders live, the curve grows, and the stats tick.
  - Click Stop mid-run. Expected: streaming halts and no further events arrive.
  - Check the browser console. Expected: no errors.
- [ ] **Step 3: Commit** `git commit -m "Add frontend shell and live view for sampling methods"`

---

### Task 14: Live view — beam tree and revision chain

**Files:** Modify `web/live.js`; Create `web/beamtree.js`

**Interfaces:** Consumes `beam_step` events (Task 6) and `sample_*` events with sequential indices (Task 7).
- `beamtree.js`: `renderBeamTree(svg: SVGElement, nodes: NodeEvent[])`. There is one column per step, and each node is a circle coloured by score on a red→green scale. Kept nodes are solid, pruned nodes are greyed at 30% opacity, and completed nodes have a ring.
  - Edges go parent→child.
  - Hovering a node shows the step text and score in a tooltip.
  - The tree redraws on each `beam_step`, and the container scrolls horizontally.
- Revisions: the grid becomes a horizontal **chain** of cards with arrows; each card shows its answer, ✓/✗ and PRM.

- [ ] **Step 1: Implement.**
- [ ] **Step 2: Verify in the browser** (mock backend).
  - Run Beam N=16. Expected: the tree grows one column per step, with 4 kept nodes per step and the others greyed.
  - Run Revisions N=8. Expected: cards appear one after another, left→right.
- [ ] **Step 3: Commit** `git commit -m "Add beam tree and revision chain visualizations"`

---

### Task 15: Bench view

**Files:** Create `web/bench.js`; Modify `web/index.html`

**Interfaces:** Consumes the Task 12 bench routes.
- Form fields:
  - questions per level (10; options 2/10/20/100)
  - seed (0)
  - method checkboxes (pool = majority + BoN, beam, revisions)
  - pool size (64)
  - beam budgets (4, 16)
  - revision budgets (4, 8)
  - a "reference-model run" checkbox, which hides the other methods and sets pool size to 8
  - [Run]
- Run list: past runs with status, lock summary, Resume (shows the 409 message and keys), and Stop.
- Progress: a progress bar plus `done/total · ETA`, polling `metrics` every 2 s while `status == "running"`.
- **Chart**: accuracy vs budget, log₂ x-axis, one line per method. pass@1 is a dashed horizontal baseline, and compute-optimal is a thick line.
  - Optional overlay: pick a finished reference run, whose pass@1 is drawn as a dotted horizontal line labelled with its `policy_model`.
  - The overlay is only allowed when neither run or both runs are mock.
- **Level table**: level × method at a selected budget, with the best value per row in bold.
- A cost table (mean tokens and seconds per question, per method), export buttons (CSV/JSON links), and a red `MOCK DATA` banner on mock runs.

- [ ] **Step 1: Implement.**
- [ ] **Step 2: Verify in the browser** (mock backend). Run a bench with 2 per level, pool 8, beam [4], revisions [2].
  - Expected: the progress bar advances, then the chart shows majority, BoN, beam, revisions, a dashed baseline and compute-optimal; the table fills; CSV downloads; and the MOCK banner is shown.
  - Stop the server mid-run, restart it, and click Resume. Expected: it completes without redoing finished items.
- [ ] **Step 3: Commit** `git commit -m "Add bench view with scaling chart, level table and export"`

---

### Task 16: Colab notebook

**Files:** Create `colab/build_notebook.py`, `colab/tts_server.ipynb` (generated); Test `tests/test_notebook.py`

**Interfaces:** `build_notebook.py` reads `colab/gateway.py` and writes the notebook (nbformat 4 JSON, written by hand; no nbformat dependency). The cells:
1. Markdown: what this is, "Runtime → GPU (A100 or L4 recommended)", and how to paste the URL/token.
2. Config: `POLICY_MODEL = "Qwen/Qwen3-1.7B"`, `PRM_MODEL = "Qwen/Qwen2.5-Math-PRM-7B"`, `GATEWAY_PORT = 9000`, `VLLM_PORT = 8001`.
3. `!pip install -q vllm==<pinned> bitsandbytes fastapi uvicorn httpx`. Pin the latest vLLM release at implementation time (PyPI showed 0.34.0 on 2026-11-17) and record it in a comment.
4. `%%writefile gateway.py` + the verbatim contents of `colab/gateway.py`.
5. Detect the GPU with `torch.cuda`:
   - `total_gb`, name, capability
   - `dtype = "half" if capability < (8,0) else "bfloat16"`
   - `prm_quant = "bf16" if total_gb >= 40 else "4bit"`
   - `util = 0.35 if total_gb >= 40 else 0.30`
   - print the tier.
6. Start vLLM: `subprocess.Popen([sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", POLICY_MODEL, "--port", str(VLLM_PORT), "--dtype", dtype, "--gpu-memory-utilization", str(util), "--max-model-len", "12288"])`, then poll `/health` for up to 10 min, tailing the log on failure.
7. Load `QwenPrm(PRM_MODEL, prm_quant)` and the policy tokenizer; `token = secrets.token_urlsafe(24)`; build `info`; run uvicorn `create_app(...)` in a daemon thread on `GATEWAY_PORT`.
8. Download `cloudflared-linux-amd64` from GitHub releases (latest), run `tunnel --url http://localhost:9000`, regex `https://[a-z0-9-]+\.trycloudflare\.com` from its stderr, and print **URL** and **TOKEN** in large text.
9. Keep-alive loop printing health every 60 s.

- [ ] **Step 1: Write failing test**

```python
def test_notebook_embeds_current_gateway():
    nb = json.loads(Path("colab/tts_server.ipynb").read_text(encoding="utf-8"))
    src = "".join("".join(c["source"]) for c in nb["cells"])
    assert Path("colab/gateway.py").read_text(encoding="utf-8").strip() in src
    assert "trycloudflare" in src and "Qwen/Qwen3-1.7B" in src and nb["nbformat"] == 4
```
- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement** and run `python colab/build_notebook.py`. **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `git commit -m "Add Colab notebook generator and notebook"`

---

### Task 17: README usage and end-to-end check

**Files:** Modify `README.md` (add "Running it" — venv setup, start command, Colab steps, mock mode, what each method is, and the deviations from the paper: off-the-shelf PRM, prompted revisions, MATH levels as oracle difficulty).

- [ ] **Step 1: Write the README section.**
- [ ] **Step 2: Full suite.** Run `pytest -v`. Expected: all PASS.
- [ ] **Step 3: Mock end-to-end.** Repeat the Task 13–15 browser checks once more on a fresh `runs/`.
- [ ] **Step 4: Colab end-to-end (user-run).** Upload `colab/tts_server.ipynb` to Colab Pro (A100 or L4) → Run all → paste URL + token → the backend bar shows Qwen3-1.7B and the PRM. Then:
  - Live: BoN N=16 on a Level-3 question.
  - Bench: 4 per level, pool 16, beam [4], revisions [4].
  - Expected: pass@1 lands in a plausible range for Qwen3-1.7B, and majority/BoN rise with N.
  - Record any version fixes the notebook needed.
- [ ] **Step 5: Commit** `git commit -m "Document running the visualizer locally and on Colab"`
