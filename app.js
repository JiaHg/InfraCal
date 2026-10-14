// Infra_cal · UI 层（Emerald Editorial）：收集输入 → 调用 engine.js → 渲染。不做数值推导。
import { MODELS, GPUS, PRECISIONS, KV_DTYPES, DEFAULT_ASSUMPTIONS, VLLM_DOCS } from "./data.js";
import * as E from "./engine.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v, d = 1) => {
  if (v == null || !Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a >= 1e9) return (v / 1e9).toFixed(2) + "B";
  if (a >= 1e6) return (v / 1e6).toFixed(2) + "M";
  if (a >= 10000) return Math.round(v).toLocaleString("en-US");
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(d);
  if (a >= 1) return v.toFixed(Math.max(d, 2));
  return v.toPrecision(3);
};

// ---------------------------------------------------------------------------
// 状态
// ---------------------------------------------------------------------------
const MODEL_DEFAULTS = {
  "llama-3.3-70b": { precAttn: "bf16", precFfn: "bf16" },
  "deepseek-v3": { precAttn: "fp8", precFfn: "fp8" },
};
function freshState() {
  return {
    modelId: "deepseek-v3", gpuId: "A100", precAttn: "fp8", precFfn: "fp8", kvDtype: "auto",
    promptLen: 2048, outputLen: 512, maxModelLen: 8192, targetConcurrency: 64, tpotSlaMs: 50,
    objective: "throughput", maxNodes: 4, assumptions: { ...DEFAULT_ASSUMPTIONS }, selected: null, pin: null,
  };
}
let state = freshState();
try {
  const h = decodeURIComponent(location.hash.slice(1));
  if (h) { const s = JSON.parse(h); state = { ...freshState(), ...s, assumptions: { ...DEFAULT_ASSUMPTIONS, ...(s.assumptions || {}) } }; }
} catch { /* 无有效状态则用默认值 */ }

function toInput() {
  return E.defaultInput({
    modelId: state.modelId, gpuId: state.gpuId, prec: { attn: state.precAttn, ffn: state.precFfn }, kvDtype: state.kvDtype,
    promptLen: +state.promptLen || 1, outputLen: +state.outputLen || 1, maxModelLen: +state.maxModelLen || 4096,
    targetConcurrency: +state.targetConcurrency || 0, tpotSlaMs: +state.tpotSlaMs || 0,
    objective: state.objective, maxNodes: Math.max(1, Math.min(8, +state.maxNodes || 1)), assumptions: { ...state.assumptions },
  });
}

let timer = null;
function scheduleUpdate() { clearTimeout(timer); timer = setTimeout(update, 180); }

// ---------------------------------------------------------------------------
// 数字滚动
// ---------------------------------------------------------------------------
const prevNums = new Map();
function countUp(el, key, target, render) {
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const from = prevNums.get(key);
  prevNums.set(key, target);
  if (reduced || from == null || !Number.isFinite(target) || from === target) { render(target); return; }
  const t0 = performance.now(), dur = 380;
  const step = (t) => {
    const p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 3);
    render(from + (target - from) * e);
    if (p < 1 && prevNums.get(key) === target) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// ---------------------------------------------------------------------------
// 主流程
// ---------------------------------------------------------------------------
let current = null;
function update() {
  const input = toInput();
  try { history.replaceState(null, "", "#" + encodeURIComponent(JSON.stringify({ ...state, pin: undefined }))); } catch { /* 沙箱 */ }
  const search = E.searchConfigs(input);
  const minNodes = E.minNodesSingleReplica(input);
  let sel = search.ranked.find((r) => r.label === state.selected) || search.ranked[0] || null;
  if (sel && state.pin && state.pin.label === sel.label) {
    const b = Math.max(1, Math.min(sel.feas.maxConc, state.pin.batch));
    sel = { ...sel, batch: b, decode: E.decodeStep(input, sel.par, sel.feas, b), pinned: true };
    sel.meetsSla = !input.tpotSlaMs || sel.decode.tpotMs <= input.tpotSlaMs;
  }
  current = { input, search, minNodes, sel };
  renderCover(); renderControls(); renderKpis(); renderVram(); renderPhase();
  renderBatch(); renderPareto(); renderPlans(); renderTopo(); renderCommands();
  renderAdv(); renderTrace(); renderSources();
}
function select(label) { state.selected = label; state.pin = null; update(); }

// ---------------------------------------------------------------------------
// 封面
// ---------------------------------------------------------------------------
function renderCover() {
  const { input, search, minNodes, sel } = current;
  const m = MODELS[input.modelId], g = GPUS[input.gpuId];
  $("modelBtn").textContent = m.name.replace("-Instruct", "");
  const nodes = sel?.par.nodes ?? minNodes ?? input.maxNodes;
  $("gpuBtn").innerHTML = `${nodes} × ${g.gpusPerNode} · ${esc(g.id)}`;
  $("mastRight").textContent = `${g.id} · ${new Date().getFullYear()}`;
  const v = $("verdict");
  if (!search.ranked.length) v.innerHTML = `<span class="p">✕ ${input.maxNodes} 台内放不下</span>${minNodes ? `　节点调到 ${minNodes} 台以上` : "　降精度或减 max_model_len"}`;
  else if (minNodes > 1) v.innerHTML = `单机放不下 — <span class="x">至少 ${minNodes} 台</span>`;
  else v.innerHTML = `<span class="x">单机可跑</span>`;
}

// ---------------------------------------------------------------------------
// 参数行
// ---------------------------------------------------------------------------
function pills(el, options, value, onPick) {
  el.innerHTML = options.map((o) => `<button type="button" data-v="${o.v}" aria-pressed="${o.v === value}" ${o.disabled ? "disabled" : ""} title="${esc(o.title || "")}">${esc(o.label)}</button>`).join("");
  el.onclick = (e) => { const b = e.target.closest("button"); if (b && !b.disabled) onPick(b.dataset.v); };
}
function renderControls() {
  const g = GPUS[state.gpuId];
  const prec = state.precAttn === state.precFfn ? state.precAttn : null;
  pills($("precPick"), Object.values(PRECISIONS).map((p) => ({
    v: p.id, label: p.id.toUpperCase(),
    disabled: p.id === "fp4" && !g.supports.fp4Compute,
    title: p.id === "fp4" && !g.supports.fp4Compute ? "A100 无 FP4 张量核" : p.label,
  })), prec, (v) => { state.precAttn = state.precFfn = v; state.pin = null; update(); });
  pills($("objPick"), [["throughput", "吞吐"], ["latency", "时延"], ["cost", "省卡"]].map(([v, label]) => ({ v, label })), state.objective, (v) => { state.objective = v; state.selected = null; state.pin = null; update(); });
  for (const id of ["promptLen", "outputLen", "targetConcurrency", "tpotSlaMs", "maxNodes"]) {
    const el = $(id);
    if (document.activeElement !== el) el.value = state[id];
    el.oninput = () => { state[id] = el.value === "" ? 0 : +el.value; state.pin = null; scheduleUpdate(); };
  }
  const warn = g.id === "A100" && (state.precAttn === "fp8" || state.precFfn === "fp8");
  $("precWarn").hidden = !warn;
  if (warn) $("precWarn").innerHTML = `A100 上 FP8 = W8A16 Marlin：省显存，不加速计算 <a href="${GPUS.A100.sources.fp8.url}" target="_blank" rel="noopener">↗</a>`;
}
$("modelBtn").onclick = () => {
  const ids = Object.keys(MODELS);
  state.modelId = ids[(ids.indexOf(state.modelId) + 1) % ids.length];
  Object.assign(state, MODEL_DEFAULTS[state.modelId]);
  fixPrec(); state.selected = null; state.pin = null; update();
};
$("gpuBtn").onclick = () => {
  const ids = Object.keys(GPUS);
  state.gpuId = ids[(ids.indexOf(state.gpuId) + 1) % ids.length];
  fixPrec(); state.selected = null; state.pin = null; update();
};
function fixPrec() {
  if (!GPUS[state.gpuId].supports.fp4Compute) { if (state.precAttn === "fp4") state.precAttn = "fp8"; if (state.precFfn === "fp4") state.precFfn = "fp8"; }
}

// ---------------------------------------------------------------------------
// 指标
// ---------------------------------------------------------------------------
function renderKpis() {
  const { input, search, sel } = current;
  const nav = $("planNav"), el = $("kpis");
  if (!sel) { nav.innerHTML = ""; el.innerHTML = `<div class="empty" style="grid-column:1/-1">无可行方案</div>`; $("kpiSide").textContent = ""; return; }
  const p = sel.par, d = sel.decode;
  $("kpiSide").textContent = `瓶颈 · ${d.bottleneck}`;
  const top = search.ranked.slice(0, 3);
  nav.innerHTML = top.map((r, i) => `<button type="button" aria-pressed="${r.label === sel.label}" data-l="${esc(r.label)}">${i + 1} — ${esc(r.label)}</button>`).join("")
    + `<span class="tagp" title="张量并行 ${p.tp} · 流水线 ${p.pp} · 数据并行 ${p.dp} · 专家并行 ${p.epEnabled ? p.ep : "关"}">TP${p.tp} · PP${p.pp} · DP${p.dp} · EP${p.epEnabled ? p.ep : "—"}</span>`;
  nav.onclick = (e) => { const b = e.target.closest("button"); if (b) select(b.dataset.l); };
  const tiles = [
    { key: "tpot", l: "TPOT", v: d.tpotMs, u: "ms", d1: `batch ${sel.batch} · b=1 时 ${fmt(sel.decode1.tpotMs)} ms`, bad: !sel.meetsSla ? "✕ 超上限" : null, tr: "TPOT" },
    { key: "tps", l: "每卡吞吐", v: d.tokPerSecPerGpu, u: "tok/s", d1: `合计 ${fmt(d.tokPerSec)} · ${p.gpus} 卡`, tr: "系统输出吞吐" },
    { key: "ttft", l: "TTFT", v: sel.prefill.ttftMs, u: "ms", d1: `输入 ${input.promptLen} tok`, tr: "TTFT" },
    { key: "cap", l: "并发容量", v: sel.capacity, u: "seqs", d1: `${p.dp} 副本 × ${fmt(sel.feas.maxConc)}`, bad: !sel.meetsConcurrency ? "✕ 不足" : null, tr: "最大并发" },
  ];
  el.innerHTML = tiles.map((t, i) => `<div class="kpi ${i % 2 ? "t-paper" : "t-ink"}">
    <span class="l">${t.l}<button class="srcb" type="button" data-trace="${esc(t.tr)}" title="计算轨迹" style="border:0;background:none;cursor:pointer;color:inherit;font-weight:800">→</button></span>
    <span class="v" data-key="${t.key}"></span>
    ${t.bad ? `<span class="bad">${t.bad}</span>` : ""}
    <span class="d">${t.d1}</span></div>`).join("");
  const vals = Object.fromEntries(tiles.map((t) => [t.key, t]));
  el.querySelectorAll(".v[data-key]").forEach((v) => {
    const t = vals[v.dataset.key];
    countUp(v, t.key, t.v, (x) => { v.innerHTML = `${fmt(x)}<small>${t.u}</small>`; });
  });
  el.querySelectorAll("[data-trace]").forEach((b) => b.onclick = () => jumpTrace(b.dataset.trace));
}

// ---------------------------------------------------------------------------
// tooltip
// ---------------------------------------------------------------------------
const tip = $("tip");
function showTip(e, html) { tip.innerHTML = html; tip.hidden = false; const x = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8); const y = Math.min(e.clientY + 14, innerHeight - tip.offsetHeight - 8); tip.style.left = x + "px"; tip.style.top = y + "px"; }
function hideTip() { tip.hidden = true; }
const kv = (k, v) => `<div class="kv"><span>${k}</span><span>${v}</span></div>`;
function niceTicks(min, max, n = 5) {
  const span = max - min || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((s) => span / s <= n) || 10 * mag;
  const lo = Math.floor(min / step) * step, hi = Math.ceil(max / step) * step, t = [];
  for (let v = lo; v <= hi + step / 2; v += step) t.push(+v.toFixed(10));
  return t;
}

// ---------------------------------------------------------------------------
// 显存（反白面板，三段：权重 / 开销 / KV）
// ---------------------------------------------------------------------------
function renderVram() {
  const { input, sel, search } = current;
  const par = sel?.par || search.infeasible.sort((a, b) => b.par.gpus - a.par.gpus)[0]?.par;
  const panel = $("vramPanel");
  if (!par) { panel.innerHTML = ""; $("memSide").textContent = ""; return; }
  const bd = E.memoryBreakdown(input, par);
  const g = GPUS[input.gpuId];
  $("memSide").textContent = `每卡 ${g.memGB} GB · 上限 ${Math.round(input.assumptions.gpuMemoryUtilization * 100)}%`;
  const merge = (s) => {
    const get = (k) => s.parts.find((p) => p.key === k)?.gb ?? 0;
    return [
      { k: "权重", gb: get("attn") + get("ffn") + get("other"), c: "var(--bg-3)", tc: "var(--ink)" },
      { k: "开销", gb: get("act") + get("overhead"), c: "var(--ink-3)", tc: "var(--paper)" },
      { k: "KV Cache", gb: get("kv"), c: "var(--bg)", tc: "var(--ink)" },
    ];
  };
  panel.innerHTML = `<div class="p-head"><b>${esc(E.parLabel(par))}</b><span>${sel ? "" : "不可行示意"}</span></div>` + bd.map((s, i) => {
    const parts = merge(s);
    const segs = parts.filter((p) => p.gb > 0.05).map((p) => {
      const w = p.gb / s.totalGB * 100;
      return `<i data-s="${i}" data-k="${esc(p.k)}" style="width:${w.toFixed(2)}%;background:${p.c}">${w > 13 ? `<em style="color:${p.tc}">${esc(p.k)} ${fmt(p.gb)}</em>` : ""}</i>`;
    }).join("");
    const used = parts.reduce((a, p) => a + p.gb, 0);
    return `<div class="vrow"><span class="rl">${bd.length > 1 ? "S" + s.stage : "每卡"}</span>
      <div><div class="vbar">${segs}</div>
      <p class="vnote">${s.overGB > 0 ? `✕ 超出 <b>${fmt(s.overGB)} GB</b>` : `<b>${fmt(used)}</b> / ${s.totalGB} GB　·　KV ≈ <b>${fmt(s.kvTokens)}</b> tokens${i === 0 && sel ? `　·　并发上限 <b>${fmt(sel.feas.maxConc)}</b>/副本` : ""}`}</p></div></div>`;
  }).join("");
  panel.querySelectorAll(".vbar i").forEach((seg) => {
    const s = bd[+seg.dataset.s], parts = merge(s), p = parts.find((x) => x.k === seg.dataset.k);
    seg.onmousemove = (e) => showTip(e, `<b>${esc(p.k)}</b>${kv("每卡", fmt(p.gb) + " GB")}${kv("占比", (p.gb / s.totalGB * 100).toFixed(1) + "%")}${p.k === "KV Cache" ? kv("tokens", fmt(s.kvTokens)) : ""}`);
    seg.onmouseleave = hideTip;
  });
}

// ---------------------------------------------------------------------------
// 时间：Prefill / Decode 对照
// ---------------------------------------------------------------------------
function renderPhase() {
  const { input, sel } = current;
  const el = $("phase");
  if (!sel) { el.innerHTML = ""; $("phaseSide").textContent = ""; return; }
  const g = GPUS[input.gpuId], A = input.assumptions, d = sel.decode, pf = sel.prefill;
  const ef = E.effectiveFlops(input.gpuId, input.prec);
  const bwEff = g.memBandwidthGBps * A.bandwidthEfficiency;
  const readGB0 = d.stages[0].memMs / 1000 * bwEff;
  const achievedBW = readGB0 / (d.tpotMs / 1000);
  const commD = d.tpMs + d.epMs + d.ppMs;
  $("phaseSide").textContent = `访存 ${fmt(d.memMs)} / 计算 ${fmt(d.compMs)} / 通信 ${fmt(commD)} ms`;
  const row = (l, v, tr) => `<div class="r"><dt>${l}${tr ? ` <button type="button" data-trace="${esc(tr)}">→</button>` : ""}</dt><span class="fig">${v}</span></div>`;
  el.innerHTML = `
    <div><h3>Prefill <span>算力受限</span></h3><div class="rows">
      ${row("计算", fmt(pf.compMs) + " ms", "Stage 0 计算量")}
      ${row("通信", fmt(pf.commMs) + " ms")}
      ${row("TTFT", fmt(pf.ttftMs) + " ms", "TTFT")}
      ${row("有效算力", `${fmt(ef.peak * A.mfuPrefill)} / ${fmt(ef.peak)} T`, "算力")}
    </div></div>
    <div><h3>Decode <span>${d.bottleneck}受限</span></h3><div class="rows">
      ${row("每步读取", fmt(readGB0) + " GB", "每步读取字节")}
      ${row("通信", fmt(commD) + " ms", "AllReduce")}
      ${row("TPOT", fmt(d.tpotMs) + " ms", "TPOT")}
      ${row("达成带宽", `${fmt(achievedBW)} / ${fmt(g.memBandwidthGBps)} GB/s`, "访存带宽")}
    </div></div>`;
  el.querySelectorAll("[data-trace]").forEach((b) => b.onclick = () => jumpTrace(b.dataset.trace));
}

// ---------------------------------------------------------------------------
// 扫描：柱=吞吐（点击钉选），线=TPOT
// ---------------------------------------------------------------------------
function renderBatch() {
  const { input, sel } = current;
  const panel = $("batchPanel");
  if (!sel) { panel.innerHTML = ""; return; }
  const pts = E.batchCurve(input, sel.par);
  const W = 960, H = 300, m = { l: 62, r: 62, t: 30, b: 40 };
  const n = pts.length, bw = (W - m.l - m.r) / n;
  const yT = niceTicks(0, Math.max(...pts.map((p) => p.tokPerSecPerGpu)) * 1.06, 4);
  const sla = input.tpotSlaMs;
  const yL = niceTicks(0, Math.max(...pts.map((p) => p.tpotMs), sla || 0) * 1.06, 4);
  const X = (i) => m.l + i * bw + bw / 2;
  const YT = (v) => H - m.b - v / yT.at(-1) * (H - m.t - m.b);
  const YL = (v) => H - m.b - v / yL.at(-1) * (H - m.t - m.b);
  const opIdx = pts.reduce((best, p, i) => Math.abs(p.batch - sel.batch) < Math.abs(pts[best].batch - sel.batch) ? i : best, 0);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${YL(p.tpotMs).toFixed(1)}`).join("");
  panel.innerHTML = `<div class="p-head"><b>batch 1 – ${fmt(sel.feas.maxConc)}</b><span>柱 吞吐 / 线 TPOT${sel.pinned ? " · 已钉选" : ""}</span></div>
  <div style="overflow-x:auto"><svg class="chart" viewBox="0 0 ${W} ${H}" style="min-width:560px" role="img" aria-label="Batch 扫描">
    <g class="grid">${yT.map((v) => `<line x1="${m.l}" x2="${W - m.r}" y1="${YT(v)}" y2="${YT(v)}"/>`).join("")}</g>
    ${yT.map((v) => `<text class="fig" x="${m.l - 9}" y="${YT(v) + 4}" text-anchor="end">${fmt(v, 0)}</text>`).join("")}
    ${yL.map((v) => `<text class="fig" x="${W - m.r + 9}" y="${YL(v) + 4}" text-anchor="start" opacity=".8">${fmt(v, 0)}</text>`).join("")}
    <text x="${m.l}" y="${m.t - 12}" text-anchor="start">TOK/S 每卡</text>
    <text x="${W - m.r}" y="${m.t - 12}" text-anchor="end">TPOT MS</text>
    ${pts.map((p, i) => `<g class="bcol" data-i="${i}" style="cursor:pointer">
      <rect x="${(m.l + i * bw + bw * 0.16).toFixed(1)}" y="${YT(p.tokPerSecPerGpu).toFixed(1)}" width="${(bw * 0.68).toFixed(1)}" height="${Math.max(1, H - m.b - YT(p.tokPerSecPerGpu)).toFixed(1)}"
        fill="${i === opIdx ? "var(--paper)" : "var(--bg)"}" opacity="${sla && p.tpotMs > sla ? .45 : 1}"/>
      <text x="${X(i)}" y="${H - m.b + 16}" text-anchor="middle" ${i === opIdx ? 'style="fill:var(--paper)"' : ""}>${p.batch}</text>
    </g>`).join("")}
    ${sla ? `<line x1="${m.l}" x2="${W - m.r}" y1="${YL(sla)}" y2="${YL(sla)}" stroke="var(--paper)" stroke-width="2"/><text x="${m.l + 6}" y="${YL(sla) - 7}" style="fill:var(--paper)">限 ${sla}</text>` : ""}
    <path d="${line}" fill="none" stroke="var(--paper)" stroke-width="3" stroke-linejoin="round" opacity=".9"/>
  </svg></div>`;
  panel.querySelectorAll(".bcol").forEach((gEl) => {
    const p = pts[+gEl.dataset.i];
    gEl.onmousemove = (e) => showTip(e, `<b>batch ${p.batch}</b>${kv("TPOT", fmt(p.tpotMs) + " ms")}${kv("每卡吞吐", fmt(p.tokPerSecPerGpu) + " tok/s")}${kv("访存/计算/通信", `${fmt(p.memMs)}/${fmt(p.compMs)}/${fmt(p.commMs)}`)}`);
    gEl.onmouseleave = hideTip;
    gEl.onclick = () => { hideTip(); state.pin = { label: sel.label, batch: p.batch }; update(); };
  });
}

// ---------------------------------------------------------------------------
// 方案：Pareto 散点 + agenda 列表
// ---------------------------------------------------------------------------
function renderPareto() {
  const { search, sel } = current;
  const pts = search.ranked, panel = $("paretoPanel");
  if (!pts.length) { panel.innerHTML = ""; return; }
  const W = 960, H = 300, m = { l: 62, r: 26, t: 30, b: 42 };
  const xs = pts.map((r) => r.decode1.tpotMs), ys = pts.map((r) => r.decode.tokPerSecPerGpu);
  const xt = niceTicks(Math.min(...xs) * 0.9, Math.max(...xs) * 1.05, 6);
  const yt = niceTicks(0, Math.max(...ys) * 1.06, 4);
  const X = (v) => m.l + (v - xt[0]) / (xt.at(-1) - xt[0]) * (W - m.l - m.r);
  const Y = (v) => H - m.b - v / yt.at(-1) * (H - m.t - m.b);
  const ok = (r) => r.meetsSla && r.meetsConcurrency;
  const order = [...pts].sort((a, b) => (a.label === sel?.label) - (b.label === sel?.label) || (a.pareto - b.pareto));
  panel.innerHTML = `<div class="p-head"><b>时延底线 × 满载吞吐</b><span>实心 Pareto / 空心 不达标 / 纸色 当前</span></div>
  <svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="方案散点">
    <g class="grid">${yt.map((v) => `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/>`).join("")}</g>
    ${yt.map((v) => `<text class="fig" x="${m.l - 9}" y="${Y(v) + 4}" text-anchor="end">${fmt(v, 0)}</text>`).join("")}
    ${xt.map((v) => `<text x="${X(v)}" y="${H - m.b + 17}" text-anchor="middle">${fmt(v, 0)}</text>`).join("")}
    <text x="${m.l}" y="${m.t - 12}">TOK/S 每卡</text>
    <text x="${W - m.r}" y="${H - m.b + 17}" text-anchor="end" opacity=".8">B=1 TPOT MS →</text>
    ${order.map((r) => {
      const isSel = r.label === sel?.label;
      const fill = isSel ? "var(--paper)" : r.pareto ? "var(--bg)" : "rgba(60,216,150,.5)";
      return `<g data-l="${esc(r.label)}" style="cursor:pointer">
        <circle cx="${X(r.decode1.tpotMs)}" cy="${Y(r.decode.tokPerSecPerGpu)}" r="15" fill="transparent"/>
        <circle cx="${X(r.decode1.tpotMs)}" cy="${Y(r.decode.tokPerSecPerGpu)}" r="${isSel ? 8 : 6}" fill="${ok(r) ? fill : "var(--ink)"}" stroke="${ok(r) ? "var(--ink)" : fill}" stroke-width="${ok(r) ? 0 : 2.5}"/>
      </g>`;
    }).join("")}
  </svg>`;
  panel.querySelectorAll("g[data-l]").forEach((gEl) => {
    const r = pts.find((x) => x.label === gEl.dataset.l);
    gEl.onmousemove = (e) => showTip(e, `<b>${esc(r.label)}</b>${kv("b=1 TPOT", fmt(r.decode1.tpotMs) + " ms")}${kv("满载 TPOT", fmt(r.decode.tpotMs) + " ms")}${kv("每卡吞吐", fmt(r.decode.tokPerSecPerGpu))}${kv("容量", r.capacity)}${ok(r) ? "" : "<div>✕ 不满足约束</div>"}`);
    gEl.onmouseleave = hideTip;
    gEl.onclick = () => { hideTip(); select(r.label); };
  });
}

let showAll = false;
function renderPlans() {
  const { search, sel } = current;
  const el = $("plans");
  $("planSide").textContent = `${search.ranked.length} 可行 · ${search.infeasible.length} 不可行`;
  if (!search.ranked.length) { el.innerHTML = `<div class="empty">无可行方案</div>`; return; }
  const rows = showAll ? search.ranked : search.ranked.slice(0, 8);
  el.innerHTML = rows.map((r, i) => `<div class="plan-r" role="button" tabindex="0" data-l="${esc(r.label)}" aria-current="${r.label === sel?.label}">
      <span class="ord">${String(i + 1).padStart(2, "0")}</span>
      <span class="name">${esc(r.label)}</span>
      <span class="fig">${fmt(r.decode.tpotMs)}<small>ms</small></span>
      <span class="fig">${fmt(r.decode.tokPerSecPerGpu)}<small>tok/s/卡</small></span>
      ${r.meetsSla && r.meetsConcurrency ? `<span class="fig">${r.par.gpus}<small>卡</small></span>` : `<span class="miss">✕ ${!r.meetsSla ? "TPOT" : "容量"}</span>`}
    </div>`).join("")
    + (search.ranked.length > 8 ? `<button class="morebtn" type="button" id="moreBtn">${showAll ? "− 收起" : `+ 其余 ${search.ranked.length - 8} 个`}</button>` : "");
  el.onclick = (e) => {
    if (e.target.closest("#moreBtn")) { showAll = !showAll; renderPlans(); return; }
    const r = e.target.closest(".plan-r"); if (r) select(r.dataset.l);
  };
  el.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { const r = e.target.closest(".plan-r"); if (r) { e.preventDefault(); select(r.dataset.l); } } };
}

// ---------------------------------------------------------------------------
// 部署
// ---------------------------------------------------------------------------
function renderTopo() {
  const { input, sel } = current;
  if (!sel) { $("topo").innerHTML = ""; $("deploySide").textContent = ""; return; }
  const p = sel.par, g = GPUS[input.gpuId], G = g.gpusPerNode, m = MODELS[input.modelId];
  $("deploySide").textContent = `${p.nodes} 台 × ${G} 卡 · ${p.nodes > 1 ? g.scaleOutGbps + " Gb/s 互联" : "NVLink " + g.nvlinkGBps + " GB/s"}`;
  const cells = [];
  let idx = 0;
  for (let rep = 0; rep < p.dp; rep++) for (let st = 0; st < p.pp; st++) for (let t = 0; t < p.tp; t++) cells.push({ rep, st, t, i: idx++ });
  const KINDS = ["g-ink", "g-paper", "g-line"];
  const kindOf = (c) => KINDS[(p.pp > 1 ? c.st : p.dp > 1 ? c.rep : 0) % 3];
  const E_ = m.config.n_routed_experts;
  $("topo").innerHTML = Array.from({ length: p.nodes }, (_, n) => {
    const cs = cells.filter((c) => Math.floor(c.i / G) === n);
    return `<div class="node"><div class="nh"><span>节点 ${n}${n === 0 ? " · HEAD" : ""}</span><span>${p.pp > 1 ? "色 = STAGE" : p.dp > 1 ? "色 = 副本" : "TP 组"}</span></div><div class="gpus">${cs.map((c) => {
      const epr = p.epEnabled ? (c.rep * p.tp + c.t) : null;
      const ex = p.epEnabled ? ` · 专家 ${epr * E_ / p.ep}–${(epr + 1) * E_ / p.ep - 1}` : "";
      return `<div class="gpu ${kindOf(c)}" title="GPU ${c.i} · R${c.rep} S${c.st} T${c.t}${ex}">${p.dp > 1 ? "R" + c.rep : p.pp > 1 ? "S" + c.st : "T" + c.t}</div>`;
    }).join("")}</div></div>`;
  }).join("");
}

let cmdCache = [];
function renderCommands() {
  const { input, sel } = current;
  if (!sel) { $("cmds").innerHTML = ""; return; }
  const { cmds, notes } = E.vllmCommands(input, sel);
  cmdCache = cmds;
  $("cmds").innerHTML = cmds.map((c, i) => `<div class="cmdt"><span>${esc(c.title)}</span><button type="button" data-copy="${i}">复制</button></div><pre class="cmd">${esc(c.cmd)}</pre>`).join("")
    + (notes.length ? `<p class="cmdnote">${notes.map((n) => esc(n.replace(/https?:\/\/\S+/g, "")).trim()).filter(Boolean).join("　·　")}</p>` : "")
    + (cmds.length > 1 ? `<p class="cmdnote">HEAD_IP = 节点 0 地址</p>` : "");
  $("cmds").onclick = (e) => { const b = e.target.closest("[data-copy]"); if (b) copy(cmdCache[+b.dataset.copy].cmd, "命令已复制"); };
}

// ---------------------------------------------------------------------------
// 假设抽屉
// ---------------------------------------------------------------------------
function renderAdv() {
  const ag = $("advGrid");
  const ADV = [
    ["gpuMemoryUtilization", "gpu_memory_utilization", 0.01],
    ["maxNumBatchedTokens", "max_num_batched_tokens", 256],
    ["nonTorchOverheadGB", "非 Torch 开销 GB", 0.1],
    ["cudaGraphGB", "CUDA Graph GB", 0.1],
    ["bandwidthEfficiency", "带宽系数 η", 0.05],
    ["mfuDecode", "Decode MFU", 0.05],
    ["mfuPrefill", "Prefill MFU", 0.05],
    ["allreduceLatencyMsNvlink", "NVLink α ms", 0.01],
    ["allreduceLatencyMsIb", "跨机 α ms", 0.05],
  ];
  if (!ag.dataset.built) {
    ag.innerHTML = `<div class="field">max_model_len<input id="maxModelLen" type="number" min="256" step="1024"></div>
      <div class="field">KV Cache<span class="pillrow" id="kvPick" style="border:4px solid var(--ink)"></span></div>
      <div class="field">注意力精度<span class="pillrow" id="pAttn" style="border:4px solid var(--ink)"></span></div>
      <div class="field">FFN/专家精度<span class="pillrow" id="pFfn" style="border:4px solid var(--ink)"></span></div>`
      + ADV.map(([k, label]) => `<div class="field">${esc(label)}<input id="a_${k}" type="number" step="${ADV.find((a) => a[0] === k)[2]}"></div>`).join("");
    ag.dataset.built = "1";
    $("maxModelLen").oninput = (e) => { state.maxModelLen = +e.target.value || 0; state.pin = null; scheduleUpdate(); };
    for (const [k] of ADV) $("a_" + k).oninput = (e) => { const v = parseFloat(e.target.value); if (Number.isFinite(v) && v > 0) { state.assumptions[k] = v; scheduleUpdate(); } };
  }
  if (document.activeElement !== $("maxModelLen")) $("maxModelLen").value = state.maxModelLen;
  for (const [k] of ADV) { const el = $("a_" + k); if (document.activeElement !== el) el.value = state.assumptions[k]; }
  pills($("kvPick"), Object.values(KV_DTYPES).map((k) => ({ v: k.id, label: k.id === "auto" ? "BF16" : "FP8" })), state.kvDtype, (v) => { state.kvDtype = v; state.pin = null; update(); });
  const g = GPUS[state.gpuId];
  const precOpts = Object.values(PRECISIONS).map((p) => ({ v: p.id, label: p.id.toUpperCase(), disabled: p.id === "fp4" && !g.supports.fp4Compute }));
  pills($("pAttn"), precOpts, state.precAttn, (v) => { state.precAttn = v; state.pin = null; update(); });
  pills($("pFfn"), precOpts, state.precFfn, (v) => { state.precFfn = v; state.pin = null; update(); });
  $("calBtn").onclick = () => {
    const { input, sel } = current;
    if (!sel) return;
    const r = E.calibrateBandwidth(input, sel.par, +$("calB").value || 1, +$("calT").value);
    if (!r) return;
    if (!r.ok) { $("calMsg").textContent = r.reason; return; }
    state.assumptions.bandwidthEfficiency = +r.eta.toFixed(3);
    $("calMsg").textContent = "";
    toast(`η = ${r.eta.toFixed(3)}`);
    update();
  };
}

// ---------------------------------------------------------------------------
// 轨迹
// ---------------------------------------------------------------------------
let lastTrace = null;
const openGroups = new Set(["Decode · 结果", "Prefill", "容量"]);
function renderTrace() {
  const { input, sel, search } = current;
  const tr = new E.Trace();
  E.paramSummary(MODELS[input.modelId], tr);
  E.bytesPerWeight(input.prec.attn, tr);
  if (input.prec.ffn !== input.prec.attn) E.bytesPerWeight(input.prec.ffn, tr);
  const par = sel?.par || search.infeasible[0]?.par;
  if (par) {
    E.evaluate(input, par, tr);
    if (sel?.pinned) E.decodeStep(input, par, sel.feas, sel.batch, tr);
  }
  lastTrace = { input: { ...input }, config: par ? E.parLabel(par) : null, steps: tr.steps };
  const groups = tr.byGroup();
  const q = ($("traceFilter").value || "").toLowerCase();
  $("trace").innerHTML = Object.entries(groups).map(([g, steps]) => {
    const vis = steps.filter((s) => !q || (g + s.label + s.formula).toLowerCase().includes(q));
    if (!vis.length) return "";
    const open = q || openGroups.has(g);
    return `<details class="tgroup" data-g="${esc(g)}" ${open ? "open" : ""}><summary><h4>${esc(g)} <small>${vis.length} 步</small></h4></summary>${vis.map((s) => {
      return `<div class="step" data-label="${esc(s.label)}">
        <span class="lbl">${esc(s.label)} <span class="kindtag k-${s.kind}">${s.kind === "assumption" ? "假设" : s.kind}</span></span>
        <span class="val">${fmt(s.value, 2)} <small style="font:700 10px var(--sans)">${esc(s.unit)}</small></span>
        <span class="fx">= ${esc(s.formula)}</span>
        ${Object.keys(s.inputs || {}).length ? `<span class="in">${Object.entries(s.inputs).map(([k, v]) => `${esc(k)}=${esc(typeof v === "number" ? +v.toPrecision(6) : v)}`).join(" · ")}</span>` : ""}
        <span class="src">${s.src ? linkSrc(s.src) + " · " : ""}<button type="button" data-fn="${esc(s.fn)}">engine.${esc(s.fn)}()</button></span>
      </div>`;
    }).join("")}</details>`;
  }).join("");
  $("trace").querySelectorAll("details.tgroup").forEach((d) => d.addEventListener("toggle", () => { if (d.open) openGroups.add(d.dataset.g); else openGroups.delete(d.dataset.g); }));
  $("trace").onclick = (e) => {
    const b = e.target.closest("[data-fn]"); if (!b) return;
    const step = b.closest(".step"); const ex = step.querySelector("pre");
    if (ex) { ex.remove(); return; }
    const pre = document.createElement("pre"); pre.textContent = E[b.dataset.fn]?.toString() || "(未找到)"; step.appendChild(pre);
  };
  $("traceFilter").oninput = () => renderTrace();
  $("exportTrace").onclick = (e) => { e.preventDefault(); copy(JSON.stringify(lastTrace, null, 2), "已复制"); };
}
const linkSrc = (s) => { const m = String(s).match(/https?:\/\/\S+/); if (!m) return esc(s); const url = m[0].replace(/[）)]$/, ""); const rest = String(s).replace(m[0], "").replace(/^\s*[—-]\s*/, ""); return `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(new URL(url).hostname)}</a>${rest ? " " + esc(rest) : ""}`; };
function jumpTrace(label) {
  $("traceCard").open = true;
  const steps = [...document.querySelectorAll("#trace .step")];
  const target = steps.find((s) => s.dataset.label.includes(label)) || steps.find((s) => (s.dataset.label + s.textContent).includes(label));
  if (!target) return;
  const grp = target.closest("details.tgroup");
  if (grp) { grp.open = true; openGroups.add(grp.dataset.g); }
  target.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
  target.classList.add("flash"); setTimeout(() => target.classList.remove("flash"), 1600);
}

// ---------------------------------------------------------------------------
// 来源
// ---------------------------------------------------------------------------
function renderSources() {
  if ($("sources").dataset.built) return;
  $("sources").dataset.built = "1";
  const ms = Object.values(MODELS).map((m) => `<div><b>${esc(m.name)}</b> — <a href="${m.source.url}" target="_blank" rel="noopener">config.json</a>${m.publishedSourceUrl ? ` · <a href="${m.publishedSourceUrl}" target="_blank" rel="noopener">公布参数量</a>` : ""}</div>`).join("");
  const gs = Object.values(GPUS).map((g) => `<div><b>${esc(g.name)}</b> — ${Object.entries(g.sources).map(([k, s]) => `<a href="${s.url}" target="_blank" rel="noopener" title="${esc(s.quote)}">${esc(k)}</a>`).join(" · ")}</div>`).join("");
  $("sources").innerHTML = `${ms}${gs}
    <div><b>vLLM</b> — <a href="${VLLM_DOCS.multiNode}" target="_blank" rel="noopener">多机 TP/PP</a> · <a href="${VLLM_DOCS.dataParallel}" target="_blank" rel="noopener">数据并行</a> · <a href="${VLLM_DOCS.expertParallel}" target="_blank" rel="noopener">专家并行</a> · 引擎附 9 项单元测试（参数量对照 70.6B / 671B / 37B / 14B MTP）</div>
    <div><b>边界</b> — 未建模排队、prefix caching、投机解码、chunked prefill、PD 分离；理论带宽偏乐观，拿到实测用「假设」里的校准反推 η。</div>`;
}

// ---------------------------------------------------------------------------
// 杂项
// ---------------------------------------------------------------------------
function toast(msg) { const t = $("toast"); t.textContent = msg; t.hidden = false; clearTimeout(toast.t); toast.t = setTimeout(() => t.hidden = true, 2400); }
async function copy(text, msg) {
  try { await navigator.clipboard.writeText(text); toast(msg); }
  catch { const ta = document.createElement("textarea"); ta.value = text; document.body.appendChild(ta); ta.select(); try { document.execCommand("copy"); toast(msg); } catch { toast("复制失败"); } ta.remove(); }
}
$("shareBtn").onclick = () => copy(location.href, "链接已复制");
$("resetBtn").onclick = () => { state = freshState(); prevNums.clear(); update(); toast("已恢复默认"); };

document.body.classList.add("boot");
setTimeout(() => document.body.classList.remove("boot"), 1400);
update();
