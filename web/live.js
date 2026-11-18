// Live view: stream one question through one method and watch the answer form.
import { $, api, banner, css, esc, METHODS } from "./app.js";
import { BeamTree } from "./beamtree.js";

const NOTES = {
  majority: "N independent samples; the most common answer wins.",
  bon: "N independent samples; each answer is weighted by the PRM verifier's score.",
  beam: "Step-by-step search: the PRM keeps the best N/4 partial solutions each step. N must be a multiple of 4.",
  revisions: "Each attempt sees the previous ones and tries to fix them. Prompted revisions + PRM vote (the paper fine-tuned a revision model).",
};

let ws = null;
let questions = [];
let byId = {};
let run = null;
let aggChart = null;
let tree = null;
const pending = new Map();
let rafId = 0;

export async function initLive() {
  tree = new BeamTree($("#beam-tree"), $("#beam-tip"), $("#beam-wrap"));
  try {
    questions = await api("/api/questions");
  } catch (e) {
    banner(`Could not load questions: ${e.message}`);
    return;
  }
  byId = Object.fromEntries(questions.map((q) => [q.id, q]));
  fillQuestions();
  $("#q-random").addEventListener("click", () => {
    const q = questions[Math.floor(Math.random() * questions.length)];
    $("#q-select").value = q.id;
    customMode(false);
    showQuestion(q);
  });
  $("#q-select").addEventListener("change", () => showQuestion(byId[$("#q-select").value]));
  $("#q-custom-toggle").addEventListener("click", () => customMode($("#custom-box").hidden));
  $("#method").addEventListener("change", updateNote);
  $("#run").addEventListener("click", startRun);
  $("#stop").addEventListener("click", () => send({ op: "stop" }));
  updateNote();
  showQuestion(byId[$("#q-select").value]);
  connect();
}

function fillQuestions() {
  const sel = $("#q-select");
  const levels = [...new Set(questions.map((q) => q.level))].sort();
  sel.innerHTML = levels.map((lv) => {
    const opts = questions.filter((q) => q.level === lv).map((q) => {
      const short = q.problem.replace(/\s+/g, " ").slice(0, 90);
      return `<option value="${esc(q.id)}">${esc(q.subject)} · ${esc(short)}${q.problem.length > 90 ? "…" : ""}</option>`;
    }).join("");
    return `<optgroup label="Level ${lv}">${opts}</optgroup>`;
  }).join("");
  const firstL3 = questions.find((q) => q.level === 3);
  if (firstL3) sel.value = firstL3.id;
}

function customMode(on) {
  $("#custom-box").hidden = !on;
  $("#q-select").disabled = on;
  $("#q-custom-toggle").textContent = on ? "Use dataset" : "Custom…";
  $("#question-card").hidden = on;
}

function showQuestion(q) {
  if (!q) return;
  $("#question-card").hidden = false;
  $("#q-level").textContent = q.level ? `Level ${q.level}` : "Custom";
  $("#q-subject").textContent = q.subject;
  $("#q-answer").textContent = q.answer;
  $("#q-problem").textContent = q.problem;
}

function updateNote() {
  $("#method-note").textContent = NOTES[$("#method").value];
}

// ---------- socket ----------
function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/live`);
  ws.onmessage = (m) => handle(JSON.parse(m.data));
  ws.onclose = () => { setRunning(false); setTimeout(connect, 1000); };
}

function send(msg) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    banner("Live connection to the local app is not open yet — retrying.");
    return false;
  }
  ws.send(JSON.stringify(msg));
  return true;
}

function startRun() {
  banner(null);
  const msg = { op: "run", method: $("#method").value, budget: Number($("#budget").value) };
  if (!$("#custom-box").hidden) {
    const problem = $("#custom-problem").value.trim();
    const answer = $("#custom-answer").value.trim();
    if (!problem || !answer) { banner("A custom question needs both a problem and an expected answer."); return; }
    msg.custom = { problem, answer };
  } else {
    msg.question_id = $("#q-select").value;
  }
  if (send(msg)) setRunning(true);
}

function setRunning(on) {
  $("#run").disabled = on;
  $("#stop").disabled = !on;
}

// ---------- events ----------
function handle(e) {
  switch (e.type) {
    case "run_start": return onStart(e);
    case "sample_delta": return queueText(e.sample, e.text);
    case "sample_done": return onSampleDone(e);
    case "tally": return renderTally(e.entries);
    case "agg_curve": return renderCurve(e.points);
    case "stats": return renderStats(e);
    case "baseline": return renderBaseline(e);
    case "beam_step": return tree.addStep(e);
    case "run_end": return onEnd(e);
    case "error": banner(e.message); setRunning(false); markStreamingDone(); return;
    case "stopped": setRunning(false); markStreamingDone(); $("#samples-sub").textContent += " · stopped"; return;
  }
}

function onStart(e) {
  const { method, budget, question } = e;
  run = { method, budget, cards: new Map(), start: performance.now() };
  pending.clear();
  if (question.level) showQuestion(question);
  else { $("#question-card").hidden = false; showQuestion(question); }
  $("#samples-title").textContent = METHODS[method].label;
  $("#samples-sub").textContent = method === "beam" ? `N = ${budget} per step · width 4`
    : method === "revisions" ? `${budget} sequential attempts` : `${budget} parallel samples`;
  $("#final").textContent = "…"; $("#final").className = "final";
  $("#baseline").textContent = "…"; $("#baseline").className = "final small-final";
  $("#tally").innerHTML = `<div class="muted small">Waiting for the first answer…</div>`;
  $("#tally-sub").textContent = method === "majority" ? "votes" : "PRM-weighted";
  renderStats({ tokens: 0, seconds: 0 });
  resetChart();

  const box = $("#samples");
  box.className = "samples";
  box.innerHTML = "";
  const isBeam = method === "beam";
  box.hidden = isBeam;
  $("#beam-wrap").hidden = !isBeam;
  if (isBeam) { tree.reset(); return; }
  if (method === "revisions") { box.classList.add("chain"); return; }
  if (budget >= 32) box.classList.add("dense");
  for (let i = 0; i < budget; i++) card(i);
}

function card(i) {
  let c = run.cards.get(i);
  if (c) return c;
  const box = $("#samples");
  if (run.method === "revisions" && i > 0) {
    const arrow = document.createElement("div");
    arrow.className = "arrow"; arrow.textContent = "→";
    box.appendChild(arrow);
  }
  const el = document.createElement("div");
  el.className = "sample streaming";
  el.innerHTML = `<div class="s-head"><span>${run.method === "revisions" ? "attempt" : "sample"} #${i + 1}</span><span class="s-tok"></span></div>
    <div class="s-body"></div><div class="s-foot"><span class="muted small">generating…</span></div>`;
  box.appendChild(el);
  c = { el, body: el.querySelector(".s-body"), foot: el.querySelector(".s-foot"), tok: el.querySelector(".s-tok"), key: null };
  run.cards.set(i, c);
  if (run.method === "revisions") el.scrollIntoView({ block: "nearest", inline: "end" });
  return c;
}

function queueText(i, text) {
  if (!run || run.method === "beam") return;
  pending.set(i, (pending.get(i) || "") + text);
  if (!rafId) rafId = requestAnimationFrame(flush);
}

function flush() {
  rafId = 0;
  for (const [i, text] of pending) {
    const c = card(i);
    const b = c.body;
    const atBottom = b.scrollHeight - b.scrollTop - b.clientHeight < 24;
    b.append(text);
    if (atBottom) b.scrollTop = b.scrollHeight;
  }
  pending.clear();
}

function mark(correct) {
  return correct ? `<span class="mark ok" title="correct">✓</span>` : `<span class="mark no" title="wrong">✗</span>`;
}

function onSampleDone(e) {
  if (!run || run.method === "beam") return;
  if (pending.has(e.sample)) flush();
  const c = card(e.sample);
  c.el.classList.remove("streaming");
  c.key = e.key;
  c.tok.textContent = `${e.tokens} tok`;
  const prm = e.prm == null ? "" : `<span class="prm" title="PRM score of the last step">PRM <span class="prm-bar"><i style="width:${Math.round(e.prm * 100)}%"></i></span>${e.prm.toFixed(2)}</span>`;
  c.foot.innerHTML = `${mark(e.correct)}<span class="s-answer" title="${esc(e.answer ?? "(no answer)")}">${esc(e.answer ?? "(no answer)")}</span>${prm}`;
}

function markStreamingDone() {
  if (!run) return;
  for (const c of run.cards.values()) c.el.classList.remove("streaming");
}

function renderTally(entries) {
  if (!run) return;
  const weighted = run.method !== "majority";
  const max = Math.max(...entries.map((x) => (weighted ? x.weight : x.votes)), 1e-9);
  $("#tally").innerHTML = entries.slice(0, 12).map((x) => {
    const v = weighted ? x.weight : x.votes;
    const na = x.answer === "(no answer)";
    const val = weighted ? `${x.weight.toFixed(2)} · ${x.votes}×` : `${x.votes}`;
    return `<div class="t-row${na ? " na" : ""}" title="${esc(x.answer)}">
      <span class="t-ans">${x.correct ? `<span class="mark ok" title="matches the expected answer">✓</span> ` : ""}${esc(x.answer)}</span>
      <span class="t-bar"><i style="width:${(100 * v / max).toFixed(1)}%"></i></span>
      <span class="t-val">${val}</span></div>`;
  }).join("") + (entries.length > 12 ? `<div class="muted small">+${entries.length - 12} more answers</div>` : "");
}

function resetChart() {
  const color = css(METHODS[run.method].color);
  const ctx = $("#agg-chart");
  if (aggChart) aggChart.destroy();
  aggChart = new Chart(ctx, {
    type: "line",
    data: { labels: [], datasets: [{ data: [], stepped: "before", borderColor: color, borderWidth: 2,
      pointRadius: 4, pointHoverRadius: 6, pointBackgroundColor: [], pointBorderColor: css("--surface"), pointBorderWidth: 2 }] },
    options: {
      animation: false, maintainAspectRatio: false,
      plugins: { legend: { display: false },
        tooltip: { callbacks: { title: (it) => `after ${it[0].label} sample${it[0].label === "1" ? "" : "s"}`,
          label: (it) => (it.raw === 1 ? "✓ aggregate is right" : "✗ aggregate is wrong") } } },
      scales: {
        x: { title: { display: true, text: "k (completed samples)", color: css("--muted") },
          grid: { color: css("--grid") }, ticks: { color: css("--muted"), maxTicksLimit: 9 } },
        y: { min: -0.15, max: 1.15, grid: { color: css("--grid") },
          afterBuildTicks: (axis) => { axis.ticks = [{ value: 0 }, { value: 1 }]; },
          ticks: { color: css("--muted"), callback: (v) => (v === 1 ? "✓ right" : "✗ wrong") } },
      },
    },
  });
}

function renderCurve(points) {
  if (!aggChart) return;
  const ds = aggChart.data.datasets[0];
  aggChart.data.labels = points.map((p) => p.k);
  ds.data = points.map((p) => (p.correct ? 1 : 0));
  ds.pointBackgroundColor = points.map((p) => (p.correct ? css("--good") : css("--bad")));
  aggChart.update();
}

function renderStats(e) {
  $("#st-tokens").textContent = `${e.tokens.toLocaleString()} tok`;
  $("#st-secs").textContent = `${e.seconds.toFixed(1)} s`;
  $("#st-rate").textContent = e.seconds > 0 ? `${Math.round(e.tokens / e.seconds).toLocaleString()} tok/s` : "– tok/s";
}

function renderBaseline(e) {
  const el = $("#baseline");
  el.innerHTML = `${mark(e.correct)} ${esc(e.answer ?? "(no answer)")}`;
}

function onEnd(e) {
  setRunning(false);
  markStreamingDone();
  $("#final").innerHTML = `${mark(e.correct)} ${esc(e.final_answer ?? "(no answer)")}`;
  if (run && e.final_key != null) {  // same grouping as the vote (\dfrac{1}{2} == \frac{1}{2})
    for (const c of run.cards.values()) c.el.classList.toggle("won", c.key === e.final_key);
  }
  renderStats(e);
}
