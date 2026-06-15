/* Shared helpers for the RAGObserve dashboard (vanilla JS, no build step). */

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}

function esc(s) {
  if (s === null || s === undefined) return "";
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

function fmtMs(ms) {
  if (ms === null || ms === undefined) return "—";
  if (ms < 1) return "<1 ms";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function fmtCost(c) {
  if (!c) return "—";
  return `$${Number(c).toFixed(4)}`;
}

function fmtTime(epoch) {
  if (!epoch) return "—";
  return new Date(epoch * 1000).toLocaleString();
}

function fmtPct(v) {
  if (v === null || v === undefined) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function fmtNum(v, digits = 3) {
  if (v === null || v === undefined) return "—";
  return Number(v).toFixed(digits);
}

function truncate(s, n = 120) {
  s = s || "";
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function getParam(name, fallback = "") {
  return new URLSearchParams(location.search).get(name) || fallback;
}

function stageBadge(stage) {
  return `<span class="badge stage-${esc(stage)}">${esc(stage)}</span>`;
}

function statusBadge(status) {
  return `<span class="badge ${status === "error" ? "error" : "ok"}">${esc(status || "ok")}</span>`;
}

/* Populate a <select> with project names; keep ?project= in sync and reload. */
async function projectSelector(selectId, onChange) {
  const sel = document.getElementById(selectId);
  const projects = await api("/api/projects");
  const current = getParam("project", projects.length ? projects[0].name : "");
  sel.innerHTML = projects.map(p =>
    `<option value="${esc(p.name)}" ${p.name === current ? "selected" : ""}>${esc(p.name)}</option>`
  ).join("") || `<option value="">(no projects yet)</option>`;
  sel.addEventListener("change", () => {
    const u = new URL(location.href);
    u.searchParams.set("project", sel.value);
    history.replaceState(null, "", u);
    onChange(sel.value);
  });
  return current;
}

/* Keep top-nav Traces/Chunks/Metrics links carrying the current project. */
document.addEventListener("DOMContentLoaded", () => {
  const project = getParam("project");
  if (!project) return;
  document.querySelectorAll(".navlink-project").forEach(a => {
    const u = new URL(a.href);
    u.searchParams.set("project", project);
    a.href = u.toString();
  });
});
