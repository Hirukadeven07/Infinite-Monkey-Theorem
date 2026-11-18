// Bench view: start/resume control tests and read the scaling results.
import { $, api, banner, css, esc, METHODS } from "./app.js";

const ORDER = ["majority", "bon", "beam", "revisions"];
const STATUS = { running: "● running", paused: "⏸ paused", stopped: "■ stopped", done: "✓ done", error: "✗ error", created: "○ created" };

let runs = [];
let selected = null;
let metrics = null;
let ref = null;
let chart = null;
let timer = 0;

export function initBench() {
  $("#bench-form").addEventListener("submit", onSubmit);
  $("#bench-form [name=reference]").addEventListener("change", onReferenceToggle);
  $("#runs-refresh").addEventListener("click", refreshRuns);
  $("#ref-select").addEventListener("change", onRefChange);
  $("#level-budget").addEventListener("change", () => metrics && renderLevelTable(metrics));
  window.addEventListener("tab", (e) => { if (e.detail === "bench") refreshRuns(); });
  refreshRuns();
}

// ---------- form ----------
function parseList(s) {
  return String(s).split(/[\s,]+/).filter(Boolean).map(Number);
}

function onReferenceToggle(e) {
  const on = e.target.checked;
  $("#bench-methods").disabled = on;
  ["beam_budgets", "revision_budgets"].forEach((n) => { $(`#bench-form [name=${n}]`).disabled = on; });
  $("#bench-form [name=pool_size]").value = on ? 8 : 64;
}

async function onSubmit(e) {
  e.preventDefault();
  const f = new FormData(e.target);
  const reference = f.get("reference") === "on";
  const methods = reference ? ["pool"] : ["pool", "beam", "revisions"].filter((m) => f.get(`m-${m}`) === "on");
  const body = {
    per_level: Number(f.get("per_level")), seed: Number(f.get("seed")), pool_size: Number(f.get("pool_size")),
    beam_budgets: reference ? [4] : parseList(f.get("beam_budgets")),
    revision_budgets: reference ? [4] : parseList(f.get("revision_budgets")),
    methods, reference,
  };
  const err = $("#bench-form-error");
  err.hidden = true;
  if (!methods.length) { err.textContent = "Pick at least one method."; err.hidden = false; return; }
  try {
    const { run_id } = await api("/api/bench", { method: "POST", body });
    await refreshRuns();
    select(run_id);
  } catch (ex) {
    err.textContent = ex.message; err.hidden = false;
  }
}

// ---------- runs ----------
async function refreshRuns() {
  try { runs = await api("/api/runs?kind=bench"); } catch (e) { banner(e.message); return; }
  const tbody = $("#runs-table tbody");
  if (!runs.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="muted">No runs yet.</td></tr>`;
  } else {
    tbody.innerHTML = runs.map((r) => {
      const p = r.progress;
      const canResume = ["paused", "stopped", "error", "created"].includes(p.status);
      const tags = (r.config.mock ? ` <span class="badge mock">MOCK</span>` : "") +
        (r.config.reference ? ` <span class="chip">reference</span>` : "");
      return `<tr data-id="${r.id}" class="${r.id === selected ? "selected" : ""}">
        <td>${r.id}</td><td>${new Date(r.created * 1000).toLocaleString()}</td>
        <td><span class="status">${STATUS[p.status] || p.status}</span></td>
        <td>${p.done}/${p.total}</td>
        <td>${esc(r.lock.policy_model)} · ${esc(r.lock.prm_precision)}${tags}</td>
        <td class="row tight">
          <button class="btn ghost small" data-act="view">View</button>
          ${canResume ? `<button class="btn ghost small" data-act="resume">Resume</button>` : ""}
          ${p.status === "running" ? `<button class="btn ghost small" data-act="stop">Stop</button>` : ""}
        </td></tr>`;
    }).join("");
    tbody.querySelectorAll("button[data-act]").forEach((b) => b.addEventListener("click", onRunAction));
  }
  fillRefSelect();
}

async function onRunAction(e) {
  const id = Number(e.target.closest("tr").dataset.id);
  const act = e.target.dataset.act;
  try {
    if (act === "resume") {
      await api(`/api/bench/${id}/resume`, { method: "POST" });
      banner(null);
    } else if (act === "stop") {
      await api(`/api/bench/${id}/stop`, { method: "POST" });
    }
  } catch (ex) {
    const keys = ex.data && ex.data.keys ? ` (differs: ${ex.data.keys.join(", ")})` : "";
    banner(`Run #${id}: ${ex.message}${keys}`);
  }
  await refreshRuns();
  select(id);
}

function select(id) {
  selected = id;
  document.querySelectorAll("#runs-table tbody tr").forEach((tr) => tr.classList.toggle("selected", Number(tr.dataset.id) === id));
  $("#run-view").hidden = false;
  ref = null;
  fillRefSelect();
  $("#ref-select").value = "";
  loadMetrics();
}

async function loadMetrics() {
  clearTimeout(timer);
  if (selected == null) return;
  try {
    metrics = await api(`/api/bench/${selected}/metrics`);
  } catch (e) {
    banner(e.message);
    return;
  }
  renderMetrics(metrics);
  if (metrics.progress.status === "running") {
    timer = setTimeout(async () => { await refreshRuns(); loadMetrics(); }, 2000);
  }
}

function fillRefSelect() {
  const sel = $("#ref-select");
  const current = sel.value;
  const cur = runs.find((r) => r.id === selected);
  const opts = runs.filter((r) => r.config.reference && r.progress.status === "done" && r.id !== selected &&
    (!cur || Boolean(r.config.mock) === Boolean(cur.config.mock)));
  sel.innerHTML = `<option value="">none</option>` +
    opts.map((r) => `<option value="${r.id}">#${r.id} · ${esc(r.lock.policy_model)}</option>`).join("");
  sel.value = opts.some((r) => String(r.id) === current) ? current : "";
}

async function onRefChange() {
  const id = $("#ref-select").value;
  if (!id) { ref = null; }
  else {
    const m = await api(`/api/bench/${id}/metrics`);
    ref = { value: m.baseline, model: m.lock.policy_model };
  }
  if (metrics) renderChart(metrics);
}

// ---------- rendering ----------
const pct = (v) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`);

function methodsIn(m) {
  return ORDER.filter((k) => m.curves[k]);
}

function renderMetrics(m) {
  const p = m.progress;
  const lock = m.lock;
  $("#mock-banner").hidden = !m.mock;
  $("#run-title").textContent = `Run #${selected} · ${lock.policy_model} · PRM ${lock.prm_precision}` +
    (m.config.reference ? " · reference" : "");
  $("#progress-bar").style.width = `${p.total ? (100 * p.done / p.total) : 0}%`;
  const eta = p.eta_seconds != null ? ` · ETA ${fmtDuration(p.eta_seconds)}` : "";
  const failed = p.failed ? ` · ${p.failed} failed (left out of the metrics)` : "";
  $("#progress-text").textContent = `${p.done}/${p.total} items · ${STATUS[p.status] || p.status}${eta}${failed}`;
  $("#run-error").textContent = p.error ? `${p.status === "paused" ? "Paused: " : ""}${p.error}` : "";
  $("#export-csv").href = `/api/bench/${selected}/export.csv`;
  $("#export-json").href = `/api/bench/${selected}/export.json`;
  renderKpis(m);
  renderChart(m);
  fillBudgetSelect(m);
  renderLevelTable(m);
  renderCost(m);
}

function fmtDuration(s) {
  if (s < 90) return `${Math.round(s)} s`;
  if (s < 5400) return `${Math.round(s / 60)} min`;
  return `${(s / 3600).toFixed(1)} h`;
}

function bestAtMax(curve) {
  const budgets = Object.keys(curve).map(Number).sort((a, b) => a - b);
  const b = budgets[budgets.length - 1];
  return { budget: b, value: curve[b] };
}

function renderKpis(m) {
  const tiles = [`<div class="kpi"><div class="k-label">Baseline · pass@1</div><div class="k-val">${pct(m.baseline)}</div></div>`];
  for (const k of methodsIn(m)) {
    const { budget, value } = bestAtMax(m.curves[k]);
    tiles.push(`<div class="kpi"><div class="k-label"><span class="swatch" style="background:var(${METHODS[k].color})"></span>${METHODS[k].label} · N=${budget}</div><div class="k-val">${pct(value)}</div></div>`);
  }
  const co = m.compute_optimal && Object.keys(m.compute_optimal).length ? bestAtMax(m.compute_optimal) : null;
  if (co) tiles.push(`<div class="kpi"><div class="k-label"><span class="swatch" style="background:var(--ink)"></span>Compute-optimal · N=${co.budget}</div><div class="k-val">${pct(co.value)}</div></div>`);
  $("#kpis").innerHTML = tiles.join("");
}

const endLabels = {
  id: "endLabels",
  afterDatasetsDraw(c) {
    const { ctx } = c;
    ctx.save();
    ctx.font = "11px system-ui, -apple-system, 'Segoe UI', sans-serif";
    ctx.fillStyle = css("--ink-2");
    ctx.textBaseline = "middle";
    // All labels sit in the right margin at their line's final value, nudged apart so
    // converging lines never stack their names on top of each other.
    const labels = [];
    c.data.datasets.forEach((ds, i) => {
      if (!ds.endLabel || !c.isDatasetVisible(i)) return;
      const pts = c.getDatasetMeta(i).data;
      const last = pts[pts.length - 1];
      if (last) labels.push({ text: ds.endLabel, y: last.y, color: ds.borderColor });
    });
    labels.sort((a, b) => a.y - b.y);
    const gap = 14;
    for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + gap);
    const overflow = labels.length ? labels[labels.length - 1].y - c.chartArea.bottom : 0;
    if (overflow > 0) labels.forEach((l) => { l.y -= overflow; });
    const x = c.chartArea.right + 10;
    for (const l of labels) {
      ctx.fillStyle = l.color;
      ctx.fillRect(x, l.y - 4, 8, 8);
      ctx.fillStyle = css("--ink-2");
      ctx.fillText(l.text, x + 13, l.y);
    }
    ctx.restore();
  },
};

function renderChart(m) {
  const methods = methodsIn(m);
  const xs = new Set();
  methods.forEach((k) => Object.keys(m.curves[k]).forEach((b) => xs.add(Math.log2(Number(b)))));
  Object.keys(m.compute_optimal || {}).forEach((b) => xs.add(Math.log2(Number(b))));
  const xmax = Math.max(1, ...xs);
  const toPts = (curve) => Object.entries(curve).map(([b, v]) => ({ x: Math.log2(Number(b)), y: v * 100, budget: Number(b) }))
    .sort((a, b) => a.x - b.x);
  const datasets = methods.map((k) => ({
    label: METHODS[k].label, endLabel: METHODS[k].label.split(" ")[0], data: toPts(m.curves[k]),
    borderColor: css(METHODS[k].color), backgroundColor: css(METHODS[k].color),
    borderWidth: 2, pointRadius: 4, pointHoverRadius: 6, pointBorderColor: css("--surface"), pointBorderWidth: 2,
  }));
  if (m.compute_optimal && Object.keys(m.compute_optimal).length) {
    datasets.push({ label: "Compute-optimal (2-fold CV)", endLabel: "Compute-optimal", data: toPts(m.compute_optimal),
      borderColor: css("--ink"), backgroundColor: css("--ink"), borderWidth: 3, pointRadius: 3 });
  }
  if (m.baseline != null) {
    datasets.push({ label: "pass@1 baseline", data: [{ x: 0, y: m.baseline * 100, budget: 1 }, { x: xmax, y: m.baseline * 100, budget: 1 }],
      borderColor: css("--muted"), backgroundColor: css("--muted"), borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0 });
  }
  if (ref && ref.value != null) {
    datasets.push({ label: `${ref.model} pass@1 (reference)`, endLabel: ref.model,
      data: [{ x: 0, y: ref.value * 100, budget: 1 }, { x: xmax, y: ref.value * 100, budget: 1 }],
      borderColor: css("--ink-2"), backgroundColor: css("--ink-2"), borderDash: [2, 3], borderWidth: 2, pointRadius: 0 });
  }
  if (chart) chart.destroy();
  chart = new Chart($("#bench-chart"), {
    type: "line",
    data: { datasets },
    plugins: [endLabels],
    options: {
      animation: false, maintainAspectRatio: false, parsing: false,
      layout: { padding: { right: 110 } },
      interaction: { mode: "nearest", intersect: false },
      plugins: {
        legend: { position: "bottom", labels: { color: css("--ink-2"), boxWidth: 12, boxHeight: 12, usePointStyle: true } },
        tooltip: { callbacks: {
          title: (it) => `N = ${it[0].raw.budget}`,
          label: (it) => `${it.dataset.label}: ${it.raw.y.toFixed(1)}%`,
        } },
      },
      scales: {
        x: { type: "linear", min: 0, max: xmax, grid: { color: css("--grid") },
          title: { display: true, text: "generation budget N (log₂ scale)", color: css("--muted") },
          ticks: { color: css("--muted"), stepSize: 1, callback: (v) => (Number.isInteger(v) ? 2 ** v : "") } },
        y: { min: 0, max: 100, grid: { color: css("--grid") },
          title: { display: true, text: "accuracy (%)", color: css("--muted") },
          ticks: { color: css("--muted"), callback: (v) => `${v}%` } },
      },
    },
  });
}

function budgetsOf(m) {
  const s = new Set();
  methodsIn(m).forEach((k) => Object.keys(m.curves[k]).forEach((b) => s.add(Number(b))));
  return [...s].sort((a, b) => a - b);
}

function fillBudgetSelect(m) {
  const sel = $("#level-budget");
  const budgets = budgetsOf(m);
  const prev = Number(sel.value);
  sel.innerHTML = budgets.map((b) => `<option value="${b}">${b}</option>`).join("");
  const shared = budgets.filter((b) => methodsIn(m).every((k) => m.curves[k][b] != null));
  sel.value = budgets.includes(prev) ? prev : (shared.length ? shared[shared.length - 1] : budgets[budgets.length - 1]);
}

function renderLevelTable(m) {
  const budget = Number($("#level-budget").value);
  const methods = methodsIn(m).filter((k) => m.by_level.some((r) => r.method === k && r.budget === budget));
  const levels = [...new Set(m.by_level.map((r) => r.level))].sort();
  const get = (lv, k, b) => m.by_level.find((r) => r.level === lv && r.method === k && r.budget === b);
  const head = `<tr><th>Level</th><th>pass@1</th>${methods.map((k) => `<th><span class="swatch" style="background:var(${METHODS[k].color})"></span> ${METHODS[k].label}</th>`).join("")}</tr>`;
  const rows = levels.map((lv) => {
    const vals = methods.map((k) => get(lv, k, budget)?.accuracy);
    const best = Math.max(...vals.filter((v) => v != null));
    const base = get(lv, "pass@1", 1);
    return `<tr><td>${lv}${base ? ` <span class="muted">(${base.n} q)</span>` : ""}</td><td>${pct(base?.accuracy)}</td>${
      vals.map((v) => `<td class="${v != null && v === best ? "best" : ""}">${pct(v)}</td>`).join("")}</tr>`;
  });
  $("#level-table").innerHTML = methods.length ? head + rows.join("")
    : `<tr><td class="muted">No method has results at N=${budget} yet.</td></tr>`;
}

function fmtTokens(t) {
  return t >= 10000 ? `${Math.round(t / 1000)}k` : t >= 1000 ? `${(t / 1000).toFixed(1)}k` : String(t);
}

function renderCost(m) {
  const methods = ["pass@1", ...methodsIn(m)].filter((k) => m.cost[k]);
  const budgets = [...new Set(methods.flatMap((k) => Object.keys(m.cost[k]).map(Number)))].sort((a, b) => a - b);
  const head = `<tr><th>Tokens · seconds</th>${budgets.map((b) => `<th>N=${b}</th>`).join("")}</tr>`;
  const rows = methods.map((k) => `<tr><td>${k === "pass@1" ? "Single sample" : METHODS[k].label}</td>${budgets.map((b) => {
    const c = m.cost[k][b];
    return c ? `<td>${fmtTokens(c.tokens)} <span class="muted">· ${c.seconds}s</span></td>` : `<td class="muted">–</td>`;
  }).join("")}</tr>`);
  $("#cost-table").innerHTML = head + rows.join("");
}
