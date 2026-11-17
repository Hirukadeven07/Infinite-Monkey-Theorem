// Beam-search tree: one column per step, nodes coloured by PRM score (one-hue ramp),
// pruned branches faded, completed beams ringed. Rebuilt on every beam_step event.
import { css } from "./app.js";

const NS = "http://www.w3.org/2000/svg";
const COL_W = 84;
const PAD_X = 40;
const PAD_Y = 30;

function hexToRgb(h) {
  const n = parseInt(h.replace("#", ""), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function scoreColor(score, lo, hi) {
  const a = hexToRgb(lo), b = hexToRgb(hi);
  const t = Math.max(0, Math.min(1, score));
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * t));
  return `rgb(${c.join(",")})`;
}

function lastStep(text) {
  const parts = text.split("\n\n").map((s) => s.trim()).filter(Boolean);
  const s = parts[parts.length - 1] || "";
  return s.length > 300 ? s.slice(0, 300) + "…" : s;
}

export class BeamTree {
  constructor(svg, tip, wrap) {
    this.svg = svg;
    this.tip = tip;
    this.wrap = wrap;
    this.reset();
  }

  reset() {
    this.steps = [];
    this.svg.innerHTML = "";
    this.svg.setAttribute("width", 0);
    this.svg.setAttribute("height", 0);
    this.tip.hidden = true;
  }

  addStep(e) {
    this.steps[e.step] = e.nodes;
    this.draw();
    this.wrap.scrollLeft = this.wrap.scrollWidth;
  }

  layout() {
    const pos = new Map();
    const maxCount = Math.max(...this.steps.map((s) => (s ? s.length : 0)), 1);
    const gap = Math.max(12, Math.min(26, 560 / maxCount));
    const height = PAD_Y * 2 + gap * (maxCount - 1);
    this.steps.forEach((nodes, t) => {
      if (!nodes) return;
      const order = [...nodes].sort((a, b) => {
        const pa = a.parent ? pos.get(a.parent)?.y ?? 0 : 0;
        const pb = b.parent ? pos.get(b.parent)?.y ?? 0 : 0;
        return pa - pb;
      });
      const span = gap * (order.length - 1);
      const top = (height - span) / 2;
      order.forEach((n, i) => pos.set(n.id, { x: PAD_X + t * COL_W, y: top + i * gap, node: n }));
    });
    return { pos, width: PAD_X * 2 + COL_W * (this.steps.length - 1), height, r: Math.min(7, gap / 2 - 1) };
  }

  draw() {
    const { pos, width, height, r } = this.layout();
    const lo = css("--seq-lo"), hi = css("--seq-hi");
    const svg = this.svg;
    svg.innerHTML = "";
    svg.setAttribute("width", Math.max(width, 200));
    svg.setAttribute("height", height);
    const edges = document.createElementNS(NS, "g");
    const nodes = document.createElementNS(NS, "g");
    svg.append(edges, nodes);
    this.steps.forEach((_, t) => {
      const label = document.createElementNS(NS, "text");
      label.setAttribute("x", PAD_X + t * COL_W);
      label.setAttribute("y", 14);
      label.setAttribute("text-anchor", "middle");
      label.setAttribute("class", "col-label");
      label.textContent = `step ${t + 1}`;
      svg.appendChild(label);
    });
    for (const { x, y, node } of pos.values()) {
      const faded = !node.kept && !node.complete;
      if (node.parent && pos.has(node.parent)) {
        const p = pos.get(node.parent);
        const path = document.createElementNS(NS, "path");
        const mx = (p.x + x) / 2;
        path.setAttribute("d", `M${p.x},${p.y} C${mx},${p.y} ${mx},${y} ${x},${y}`);
        path.setAttribute("class", "edge" + (faded ? " pruned" : ""));
        edges.appendChild(path);
      }
      const c = document.createElementNS(NS, "circle");
      c.setAttribute("cx", x);
      c.setAttribute("cy", y);
      c.setAttribute("r", Math.max(4, r));
      c.setAttribute("fill", scoreColor(node.score, lo, hi));
      c.setAttribute("class", "node" + (faded ? " pruned" : "") + (node.complete ? " complete" : ""));
      c.addEventListener("mouseenter", () => this.showTip(node, x, y));
      c.addEventListener("mouseleave", () => { this.tip.hidden = true; });
      nodes.appendChild(c);
    }
  }

  showTip(node, x, y) {
    const state = node.complete ? "finished beam" : node.kept ? "kept" : "pruned";
    this.tip.innerHTML = "";
    const head = document.createElement("div");
    head.innerHTML = `<b>PRM ${node.score.toFixed(3)}</b> · ${state}`;
    const body = document.createElement("div");
    body.textContent = lastStep(node.text);
    this.tip.append(head, body);
    this.tip.hidden = false;
    const left = Math.min(x + 14, this.wrap.scrollWidth - 370);
    this.tip.style.left = `${Math.max(4, left)}px`;
    this.tip.style.top = `${y + 10}px`;
  }
}
