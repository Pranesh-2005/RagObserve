/* Dependency-free SVG charts for RAGObserve — no CDN, works fully offline.
 * Each helper returns an SVG string you drop into innerHTML. Colours read from
 * CSS custom properties so charts follow the theme. */

const SVGNS = "http://www.w3.org/2000/svg";

function _palette() {
  const css = getComputedStyle(document.documentElement);
  const v = n => (css.getPropertyValue(n) || "").trim();
  return {
    accent: v("--accent") || "#4f9cf9",
    green: v("--green") || "#3fb950",
    amber: v("--amber") || "#d29922",
    purple: v("--purple") || "#bc8cff",
    red: v("--red") || "#f85149",
    muted: v("--muted") || "#8b949e",
    border: v("--border") || "#2d333b",
    panel2: v("--panel2") || "#1c2128",
  };
}

const SERIES = ["accent", "purple", "green", "amber", "red"];

function _esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

/* Line chart for a single metric over time (e.g. daily cost). */
function lineChart(points, { width = 640, height = 200, color = "accent", fmt = String, label = "" } = {}) {
  const P = _palette();
  const stroke = P[color] || color;
  const pad = { l: 52, r: 16, t: 16, b: 28 };
  const w = width - pad.l - pad.r, h = height - pad.t - pad.b;
  if (!points.length) return _empty(width, height);
  const xs = points.map((_, i) => i);
  const ys = points.map(p => p.y);
  const maxY = Math.max(...ys, 0) || 1;
  const x = i => pad.l + (points.length === 1 ? w / 2 : (i / (points.length - 1)) * w);
  const y = v => pad.t + h - (v / maxY) * h;

  const line = xs.map(i => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(ys[i]).toFixed(1)}`).join(" ");
  const area = `${line} L${x(xs[xs.length - 1]).toFixed(1)},${pad.t + h} L${x(0).toFixed(1)},${pad.t + h} Z`;
  const gid = "g" + Math.random().toString(36).slice(2, 8);

  const grid = [0, 0.5, 1].map(f => {
    const gy = pad.t + h - f * h;
    return `<line x1="${pad.l}" y1="${gy}" x2="${width - pad.r}" y2="${gy}" stroke="${P.border}" stroke-width="1"/>`
      + `<text x="${pad.l - 8}" y="${gy + 3}" text-anchor="end" fill="${P.muted}" font-size="10">${_esc(fmt(maxY * f))}</text>`;
  }).join("");

  const dots = xs.map(i => `<circle cx="${x(i).toFixed(1)}" cy="${y(ys[i]).toFixed(1)}" r="2.5" fill="${stroke}">`
    + `<title>${_esc(points[i].x)}: ${_esc(fmt(ys[i]))}</title></circle>`).join("");

  const labels = points.map((p, i) => (points.length <= 12 || i % Math.ceil(points.length / 8) === 0)
    ? `<text x="${x(i).toFixed(1)}" y="${height - 8}" text-anchor="middle" fill="${P.muted}" font-size="10">${_esc(p.x)}</text>` : "").join("");

  return `<svg viewBox="0 0 ${width} ${height}" width="100%" class="chart" role="img" aria-label="${_esc(label)}">
    <defs><linearGradient id="${gid}" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0%" stop-color="${stroke}" stop-opacity="0.28"/>
      <stop offset="100%" stop-color="${stroke}" stop-opacity="0"/>
    </linearGradient></defs>
    ${grid}<path d="${area}" fill="url(#${gid})"/>
    <path d="${line}" fill="none" stroke="${stroke}" stroke-width="2" stroke-linejoin="round"/>
    ${dots}${labels}</svg>`;
}

/* Horizontal bar chart (e.g. cost by model). items: [{label, value}] */
function barChart(items, { width = 640, rowH = 30, color = "accent", fmt = String, label = "" } = {}) {
  const P = _palette();
  if (!items.length) return _empty(width, 120);
  const height = items.length * rowH + 16;
  const labelW = 150, pad = 12, barMax = width - labelW - pad - 70;
  const maxV = Math.max(...items.map(i => i.value), 0) || 1;
  const rows = items.map((it, i) => {
    const y = 8 + i * rowH;
    const bw = Math.max((it.value / maxV) * barMax, 1);
    const c = P[SERIES[i % SERIES.length]];
    return `<text x="${labelW - 8}" y="${y + rowH / 2 + 4}" text-anchor="end" fill="var(--text)" font-size="12" class="mono">${_esc(truncate(it.label, 20))}<title>${_esc(it.label)}</title></text>
      <rect x="${labelW}" y="${y + 5}" width="${bw.toFixed(1)}" height="${rowH - 12}" rx="3" fill="${c}"/>
      <text x="${labelW + bw + 6}" y="${y + rowH / 2 + 4}" fill="${P.muted}" font-size="11">${_esc(fmt(it.value))}</text>`;
  }).join("");
  return `<svg viewBox="0 0 ${width} ${height}" width="100%" class="chart" role="img" aria-label="${_esc(label)}">${rows}</svg>`;
}

/* Donut chart (e.g. token split / cost share). items: [{label, value}] */
function donutChart(items, { size = 180, fmt = String } = {}) {
  const P = _palette();
  const total = items.reduce((s, i) => s + i.value, 0);
  if (!total) return _empty(size, size);
  const cx = size / 2, cy = size / 2, r = size / 2 - 6, rin = r * 0.6;
  let a0 = -Math.PI / 2;
  const arcs = items.map((it, i) => {
    const frac = it.value / total;
    const a1 = a0 + frac * Math.PI * 2;
    const big = frac > 0.5 ? 1 : 0;
    const p = (ang, rad) => `${(cx + rad * Math.cos(ang)).toFixed(2)},${(cy + rad * Math.sin(ang)).toFixed(2)}`;
    const d = `M${p(a0, r)} A${r},${r} 0 ${big} 1 ${p(a1, r)} L${p(a1, rin)} A${rin},${rin} 0 ${big} 0 ${p(a0, rin)} Z`;
    a0 = a1;
    return `<path d="${d}" fill="${P[SERIES[i % SERIES.length]]}"><title>${_esc(it.label)}: ${_esc(fmt(it.value))} (${(frac * 100).toFixed(0)}%)</title></path>`;
  }).join("");
  return `<svg viewBox="0 0 ${size} ${size}" width="${size}" height="${size}" class="chart" role="img">
    ${arcs}<text x="${cx}" y="${cy - 2}" text-anchor="middle" fill="var(--text)" font-size="15" font-weight="700">${_esc(fmt(total))}</text>
    <text x="${cx}" y="${cy + 14}" text-anchor="middle" fill="${P.muted}" font-size="10">total</text></svg>`;
}

function legend(items, fmt = String) {
  const P = _palette();
  return `<div class="legend">${items.map((it, i) =>
    `<span class="legend-item"><span class="swatch" style="background:${P[SERIES[i % SERIES.length]]}"></span>
      ${_esc(it.label)} <span class="muted">${_esc(fmt(it.value))}</span></span>`).join("")}</div>`;
}

function _empty(w, h) {
  const P = _palette();
  return `<svg viewBox="0 0 ${w} ${h}" width="100%" class="chart"><text x="${w / 2}" y="${h / 2}" text-anchor="middle" fill="${P.muted}" font-size="12">no data yet</text></svg>`;
}
