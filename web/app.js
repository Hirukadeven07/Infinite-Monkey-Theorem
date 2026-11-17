// Shell: tabs, backend pill + dialog, shared helpers.
import { initLive } from "./live.js";
import { initBench } from "./bench.js";

export const METHODS = {
  majority: { label: "Majority vote", color: "--s1" },
  bon: { label: "Best-of-N weighted", color: "--s2" },
  beam: { label: "Beam search", color: "--s3" },
  revisions: { label: "Sequential revisions", color: "--s4" },
};

export const $ = (sel, root = document) => root.querySelector(sel);

export function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body && typeof opts.body !== "string" ? JSON.stringify(opts.body) : opts.body,
  });
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  if (!res.ok) {
    const detail = data && data.detail;
    const msg = typeof detail === "string" ? detail
      : Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : `HTTP ${res.status}`;
    const err = new Error(msg);
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}

export function banner(msg) {
  const el = $("#banner");
  if (!msg) { el.hidden = true; return; }
  el.textContent = msg;
  el.hidden = false;
}

export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
};

// ---------- backend ----------
export const backendState = { kind: "mock", connected: true, info: {} };
const listeners = [];
export function onBackend(fn) { listeners.push(fn); }

function renderBackend(b) {
  Object.assign(backendState, b);
  const dot = $("#backend-dot");
  dot.className = "dot " + (b.connected ? "on" : "off");
  const i = b.info || {};
  const text = b.kind === "mock"
    ? "Mock backend · fake model"
    : `Colab · ${i.policy_model || "?"} · PRM ${i.prm_precision || "?"} (${i.gpu || "?"})`;
  $("#backend-text").textContent = b.connected ? text : `${text} · disconnected`;
  $("#backend-mock").hidden = !(i.mock);
  listeners.forEach((fn) => fn(backendState));
}

export async function refreshBackend() {
  try { renderBackend(await api("/api/backend")); } catch (e) { banner(`Cannot reach the local app: ${e.message}`); }
}

function initBackendDialog() {
  const dlg = $("#backend-dialog");
  const url = $("#be-url"), token = $("#be-token"), err = $("#be-error");
  url.value = store.get("tts.url") || "";
  token.value = store.get("tts.token") || "";
  const kind = () => dlg.querySelector("input[name=kind]:checked").value;
  const sync = () => { const colab = kind() === "colab"; url.disabled = token.disabled = !colab; };
  dlg.querySelectorAll("input[name=kind]").forEach((r) => r.addEventListener("change", sync));
  $("#backend-pill").addEventListener("click", () => {
    err.hidden = true;
    dlg.querySelector(`input[value=${backendState.kind}]`).checked = true;
    sync();
    dlg.showModal();
  });
  $("#be-cancel").addEventListener("click", () => dlg.close());
  $("#be-connect").addEventListener("click", async () => {
    const btn = $("#be-connect");
    btn.disabled = true; btn.textContent = "Connecting…"; err.hidden = true;
    try {
      const body = kind() === "colab" ? { kind: "colab", url: url.value.trim(), token: token.value.trim() } : { kind: "mock" };
      if (body.kind === "colab") { store.set("tts.url", body.url); store.set("tts.token", body.token); }
      renderBackend(await api("/api/backend", { method: "POST", body }));
      banner(null);
      dlg.close();
    } catch (e) {
      err.textContent = e.message; err.hidden = false;
    } finally {
      btn.disabled = false; btn.textContent = "Connect";
    }
  });
}

function initTabs() {
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
    document.querySelectorAll(".tab-panel").forEach((p) => { p.hidden = p.id !== `tab-${t.dataset.tab}`; });
    window.dispatchEvent(new CustomEvent("tab", { detail: t.dataset.tab }));
  }));
}

initTabs();
initBackendDialog();
await refreshBackend();
setInterval(refreshBackend, 15000);
initLive();
initBench();
