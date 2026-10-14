// ============================================================================
// Infra_cal · 计算引擎（纯函数，无 UI 依赖，可在 Node 与浏览器中运行）
//
// 三层：
//   1) 可行性    feasibility：参数量 → 每卡权重显存 → 激活/开销 → KV 预算 → 并发容量
//   2) Roofline  estimatePerf：Decode 访存/算力/通信三者取瓶颈；Prefill 算力 + 通信
//   3) 配置搜索  searchConfigs：枚举 节点×TP×PP×DP×EP → 过滤 → 排序 → 生成 vllm serve
//
// 可追溯：所有函数都接受一个 Trace（可为 null）。每一步记录
//   { group, label, formula, inputs, value, unit, kind, src, fn }
//   kind: config（模型配置原文）| datasheet（硬件官方规格）| formula（推导）| assumption（可调假设）
// ============================================================================

import { MODELS, GPUS, PRECISIONS, KV_DTYPES, DEFAULT_ASSUMPTIONS, VLLM_DOCS } from "./data.js";

export const GB = 1e9;

// ---------------------------------------------------------------------------
// Trace
// ---------------------------------------------------------------------------
export class Trace {
  constructor() { this.steps = []; }
  add(step) { this.steps.push(step); return step.value; }
  byGroup() {
    const g = {};
    for (const s of this.steps) (g[s.group] ||= []).push(s);
    return g;
  }
}
function T(tr, step) { return tr ? tr.add(step) : step.value; }

// ---------------------------------------------------------------------------
// 1. 精度 → 每参数字节数
// ---------------------------------------------------------------------------
export function bytesPerWeight(precId, tr = null) {
  const p = PRECISIONS[precId];
  const scale = p.groupSize ? p.scaleBits / 8 / p.groupSize : 0;
  const v = p.bits / 8 + scale;
  return T(tr, {
    group: "精度", label: `${p.label} 每参数字节`, fn: "bytesPerWeight",
    formula: p.groupSize ? "bits/8 + scaleBits/8/groupSize" : "bits/8",
    inputs: { bits: p.bits, scaleBits: p.scaleBits, groupSize: p.groupSize ?? "-" },
    value: v, unit: "B/param", kind: "formula", src: p.source,
  });
}

// ---------------------------------------------------------------------------
// 2. 参数分解：把模型拆成带「切分规则」的组件
//    shard: tp（按 TP 切）| rep（每卡复制）| kv（GQA KV 头，TP>nKV 时复制）
//           vocab（词表并行，按 TP 切）| expert（开 EP 时按 EP 切，否则按 TP 切）
//    part : attn（注意力/稠密投影）| ffn（FFN / 专家）| other（embedding、norm、router）
//    quant: 是否是可量化的 Linear 权重
// ---------------------------------------------------------------------------
export function modelComponents(model) {
  const c = model.config;
  const h = c.hidden_size, V = c.vocab_size, L = c.num_hidden_layers;
  const global = [
    { name: "embed_tokens", params: V * h, shard: "vocab", part: "other", quant: false, formula: "vocab_size × hidden_size" },
    { name: "lm_head", params: V * h, shard: "vocab", part: "other", quant: false, formula: "vocab_size × hidden_size", last: true },
    { name: "final_norm", params: h, shard: "rep", part: "other", quant: false, formula: "hidden_size", last: true },
  ];
  const layers = [];
  for (let i = 0; i < L; i++) layers.push(layerComponents(model, i));
  return { global, layers };
}

export function layerComponents(model, i) {
  const c = model.config;
  const h = c.hidden_size;
  const comps = [{ name: "layernorms", params: 2 * h, shard: "rep", part: "other", quant: false, formula: "2 × hidden_size" }];
  if (model.attention === "gqa") {
    const hd = c.head_dim ?? h / c.num_attention_heads;
    const nH = c.num_attention_heads, nKV = c.num_key_value_heads;
    comps.push(
      { name: "q_proj", params: h * nH * hd, shard: "tp", part: "attn", quant: true, formula: "hidden × n_heads × head_dim" },
      { name: "k_proj+v_proj", params: 2 * h * nKV * hd, shard: "kv", part: "attn", quant: true, formula: "2 × hidden × n_kv_heads × head_dim" },
      { name: "o_proj", params: nH * hd * h, shard: "tp", part: "attn", quant: true, formula: "n_heads × head_dim × hidden" },
    );
  } else if (model.attention === "mla") {
    const nH = c.num_attention_heads, qk = c.qk_nope_head_dim + c.qk_rope_head_dim;
    comps.push(
      { name: "q_a_proj", params: h * c.q_lora_rank, shard: "rep", part: "attn", quant: true, formula: "hidden × q_lora_rank" },
      { name: "q_a_layernorm", params: c.q_lora_rank, shard: "rep", part: "other", quant: false, formula: "q_lora_rank" },
      { name: "q_b_proj", params: c.q_lora_rank * nH * qk, shard: "tp", part: "attn", quant: true, formula: "q_lora_rank × n_heads × (qk_nope+qk_rope)" },
      { name: "kv_a_proj_with_mqa", params: h * (c.kv_lora_rank + c.qk_rope_head_dim), shard: "rep", part: "attn", quant: true, formula: "hidden × (kv_lora_rank + qk_rope)" },
      { name: "kv_a_layernorm", params: c.kv_lora_rank, shard: "rep", part: "other", quant: false, formula: "kv_lora_rank" },
      { name: "kv_b_proj", params: c.kv_lora_rank * nH * (c.qk_nope_head_dim + c.v_head_dim), shard: "tp", part: "attn", quant: true, formula: "kv_lora_rank × n_heads × (qk_nope + v_head_dim)" },
      { name: "o_proj", params: nH * c.v_head_dim * h, shard: "tp", part: "attn", quant: true, formula: "n_heads × v_head_dim × hidden" },
    );
  }
  const isMoe = model.family === "moe" && i >= (c.first_k_dense_replace ?? 0) && (i % (c.moe_layer_freq ?? 1) === 0);
  if (!isMoe) {
    comps.push({ name: "mlp (gate/up/down)", params: 3 * h * c.intermediate_size, shard: "tp", part: "ffn", quant: true, formula: "3 × hidden × intermediate_size" });
  } else {
    const E = c.n_routed_experts, mI = c.moe_intermediate_size;
    comps.push(
      { name: "router gate", params: h * E + E, shard: "rep", part: "other", quant: false, formula: "hidden × n_routed_experts + n_routed_experts(bias)" },
      { name: "routed experts", params: E * 3 * h * mI, shard: "expert", part: "ffn", quant: true, expert: true, formula: "n_routed_experts × 3 × hidden × moe_intermediate" },
      { name: "shared experts", params: c.n_shared_experts * 3 * h * mI, shard: "tp", part: "ffn", quant: true, formula: "n_shared_experts × 3 × hidden × moe_intermediate" },
    );
  }
  return { index: i, isMoe, comps };
}

export function paramSummary(model, tr = null) {
  const { global, layers } = modelComponents(model);
  const c = model.config;
  let total = 0, routed = 0;
  for (const g of global) total += g.params;
  for (const l of layers) for (const x of l.comps) { total += x.params; if (x.expert) routed += x.params; }
  const k = c.num_experts_per_tok ?? 0, E = c.n_routed_experts ?? 1;
  const active = total - routed + (model.family === "moe" ? routed * k / E : 0);
  // MTP（DeepSeek num_nextn_predict_layers）：一个完整 MoE 层 + eh_proj(2h→h) + 2 个 norm + 共享 embedding/head 副本
  let mtp = 0;
  if (c.num_nextn_predict_layers) {
    const lay = layerComponents(model, c.num_hidden_layers).comps.reduce((a, x) => a + x.params, 0);
    const h = c.hidden_size;
    mtp = c.num_nextn_predict_layers * (lay + 2 * h * h + 3 * h + 2 * c.vocab_size * h);
  }
  T(tr, { group: "参数量", label: "总参数（不含 MTP）", fn: "paramSummary", formula: "Σ 各组件参数（见 modelComponents）", inputs: { layers: c.num_hidden_layers }, value: total, unit: "params", kind: "formula", src: model.source.url });
  if (model.family === "moe")
    T(tr, { group: "参数量", label: "每 token 激活参数", fn: "paramSummary", formula: "total − routed_experts + routed_experts × top_k / n_routed_experts", inputs: { total, routed, top_k: k, n_routed_experts: E }, value: active, unit: "params", kind: "formula", src: model.source.url });
  return { total, routed, active, mtp };
}

// ---------------------------------------------------------------------------
// 3. 每卡权重（按 PP stage）
// ---------------------------------------------------------------------------
export function ppLayerRanges(L, pp) {
  // 与 vLLM get_pp_indices 一致的均匀切分：余数分给靠后的 stage（前段 stage 含 embedding，较重）
  const base = Math.floor(L / pp), rem = L % pp;
  const ranges = []; let s = 0;
  for (let i = 0; i < pp; i++) { const n = base + (i >= pp - rem ? 1 : 0); ranges.push([s, s + n]); s += n; }
  return ranges;
}

function perRankParams(comp, model, par) {
  const { tp, ep, epEnabled } = par;
  switch (comp.shard) {
    case "tp": case "vocab": return comp.params / tp;
    case "rep": return comp.params;
    case "kv": { const nKV = model.config.num_key_value_heads; return comp.params / nKV * Math.max(1, nKV / tp); }
    case "expert": return epEnabled ? comp.params / ep : comp.params / tp;
  }
  throw new Error("unknown shard " + comp.shard);
}

export function precisionOfComp(comp, prec) {
  if (!comp.quant) return "bf16";
  return comp.part === "ffn" ? prec.ffn : prec.attn;
}

export function stageWeights(model, par, prec, stage, tr = null) {
  const { layers, global } = modelComponents(model);
  const [a, b] = ppLayerRanges(model.config.num_hidden_layers, par.pp)[stage];
  const bpw = { bf16: bytesPerWeight("bf16"), fp8: bytesPerWeight("fp8"), fp4: bytesPerWeight("fp4") };
  const breakdown = { attn: 0, ffn: 0, experts: 0, other: 0 };
  let expertBytes = 0, nonExpertBytes = 0, activeParams = 0, expertParamsRank = 0;
  const add = (comp) => {
    const p = perRankParams(comp, model, par);
    const bytes = p * bpw[precisionOfComp(comp, prec)];
    if (comp.expert) { breakdown.experts += bytes; expertBytes += bytes; expertParamsRank += p; }
    else { breakdown[comp.part] += bytes; nonExpertBytes += bytes; if (comp.name !== "embed_tokens") activeParams += p; }
  };
  for (const g of global) {
    if (g.last && stage === par.pp - 1) add(g);
    if (!g.last && stage === 0) add(g);
  }
  for (let i = a; i < b; i++) for (const comp of layers[i].comps) add(comp);
  const total = expertBytes + nonExpertBytes;
  if (tr) {
    const kinds = `TP=${par.tp}${model.family === "moe" ? `, ${par.epEnabled ? "EP=" + par.ep : "EP 关闭(专家按 TP 切)"}` : ""}`;
    T(tr, { group: "显存 · 权重", label: `Stage ${stage} 每卡权重（层 ${a}–${b - 1}）`, fn: "stageWeights",
      formula: "Σ_组件 每卡参数(按切分规则) × 每参数字节(按精度)", inputs: { layers: b - a, parallel: kinds, attn精度: prec.attn, ffn精度: prec.ffn },
      value: total / GB, unit: "GB", kind: "formula", src: model.source.url });
  }
  return { total, expertBytes, nonExpertBytes, breakdown, activeParams, expertParamsRank, layers: b - a, range: [a, b] };
}

// ---------------------------------------------------------------------------
// 4. KV cache 每 token 字节（每卡，按 stage 层数）
// ---------------------------------------------------------------------------
export function kvBytesPerToken(model, par, kvDtype, nLayers, tr = null) {
  const c = model.config, kb = KV_DTYPES[kvDtype].bytes;
  let v, formula, inputs;
  if (model.attention === "gqa") {
    const hd = c.head_dim ?? c.hidden_size / c.num_attention_heads;
    const kvPerRank = Math.max(1, c.num_key_value_heads / par.tp);
    v = 2 * nLayers * kvPerRank * hd * kb;
    formula = "2 × layers × max(1, n_kv_heads / TP) × head_dim × kv_bytes";
    inputs = { layers: nLayers, n_kv_heads: c.num_key_value_heads, TP: par.tp, head_dim: hd, kv_bytes: kb };
  } else {
    v = nLayers * (c.kv_lora_rank + c.qk_rope_head_dim) * kb;
    formula = "layers × (kv_lora_rank + qk_rope_head_dim) × kv_bytes   // MLA 潜向量在 TP 各卡上复制，不随 TP 切分";
    inputs = { layers: nLayers, kv_lora_rank: c.kv_lora_rank, qk_rope_head_dim: c.qk_rope_head_dim, kv_bytes: kb };
  }
  return T(tr, { group: "显存 · KV Cache", label: "每 token KV（每卡）", fn: "kvBytesPerToken", formula, inputs, value: v, unit: "bytes", kind: "formula", src: model.source.url });
}

// ---------------------------------------------------------------------------
// 5. 激活峰值（vLLM 启动时 profile 的峰值，经验估算）
// ---------------------------------------------------------------------------
export function activationPeak(model, par, A, stage, maxNumSeqs, tr = null) {
  const c = model.config, h = c.hidden_size, Tk = A.maxNumBatchedTokens, ab = A.actBytes;
  const hidden = 6 * Tk * h * ab;
  let ffn;
  if (model.family === "moe") {
    const k = c.num_experts_per_tok, mI = c.moe_intermediate_size;
    const moeRank = par.epEnabled ? mI : mI / par.tp;
    ffn = Math.max(3 * Tk * k * moeRank * ab + Tk * c.n_routed_experts * 4, 3 * Tk * c.intermediate_size / par.tp * ab);
  } else {
    ffn = 3 * Tk * c.intermediate_size / par.tp * ab;
  }
  const logits = stage === par.pp - 1 ? Math.min(Tk, maxNumSeqs) * c.vocab_size / par.tp * 4 : 0;
  const v = hidden + ffn + logits;
  return T(tr, { group: "显存 · 激活与开销", label: `Stage ${stage} 激活峰值（估算）`, fn: "activationPeak",
    formula: "6·T·h·act_bytes + FFN中间态(3·T·[top_k]·I_rank·act_bytes) + 末级 logits(min(T,max_num_seqs)·V/TP·4)",
    inputs: { T: Tk, h, TP: par.tp, max_num_seqs: maxNumSeqs, act_bytes: ab }, value: v / GB, unit: "GB", kind: "assumption",
    src: "经验公式：对应 vLLM 启动时 profile_run 的峰值；可通过「高级参数 → max_num_batched_tokens」调整" });
}

// ---------------------------------------------------------------------------
// 6. 可行性：单个并行配置
// ---------------------------------------------------------------------------
export function feasibility(input, par, tr = null) {
  const model = MODELS[input.modelId], gpu = GPUS[input.gpuId], A = input.assumptions;
  const memGB = input.memOverrideGB || gpu.memGB;
  T(tr, { group: "硬件", label: "单卡显存", fn: "feasibility", formula: "datasheet", inputs: { gpu: gpu.name }, value: memGB, unit: "GB", kind: "datasheet", src: gpu.sources.memGB.url + " — " + gpu.sources.memGB.quote });
  const util = T(tr, { group: "显存 · 激活与开销", label: "gpu_memory_utilization", fn: "feasibility", formula: "vLLM 参数", inputs: {}, value: A.gpuMemoryUtilization, unit: "", kind: "assumption", src: "vLLM 默认 0.9" });
  const budget = memGB * GB * util;
  const overhead = (A.nonTorchOverheadGB + A.cudaGraphGB) * GB;
  T(tr, { group: "显存 · 激活与开销", label: "非 PyTorch 开销 + CUDA Graph", fn: "feasibility", formula: "nonTorchOverheadGB + cudaGraphGB", inputs: { nonTorchOverheadGB: A.nonTorchOverheadGB, cudaGraphGB: A.cudaGraphGB }, value: overhead / GB, unit: "GB", kind: "assumption", src: "经验假设，可调" });
  const maxNumSeqs = input.maxNumSeqs ?? 1024;
  const stages = [];
  let capTokens = Infinity;
  for (let s = 0; s < par.pp; s++) {
    const w = stageWeights(model, par, input.prec, s, tr);
    const act = activationPeak(model, par, A, s, maxNumSeqs, tr) * GB;
    const kvpt = kvBytesPerToken(model, par, input.kvDtype, w.layers, s === 0 ? tr : null);
    const kvBudget = budget - w.total - act - overhead;
    const tokens = kvBudget > 0 ? Math.floor(kvBudget / kvpt / 16) * 16 : 0;
    T(tr, { group: "显存 · KV Cache", label: `Stage ${s} KV 可用预算`, fn: "feasibility", formula: "显存 × util − 权重 − 激活 − 开销", inputs: { 显存GB: memGB, util, 权重GB: +(w.total / GB).toFixed(2), 激活GB: +(act / GB).toFixed(2), 开销GB: overhead / GB }, value: kvBudget / GB, unit: "GB", kind: "formula" });
    T(tr, { group: "显存 · KV Cache", label: `Stage ${s} 可容纳 KV token`, fn: "feasibility", formula: "floor(KV预算 / 每token KV / block_size) × block_size", inputs: { KV预算GB: +(kvBudget / GB).toFixed(2), 每tokenKV_bytes: kvpt, block_size: 16 }, value: tokens, unit: "tokens", kind: "formula" });
    stages.push({ stage: s, weights: w, act, overhead, kvBudget, kvPerToken: kvpt, kvTokens: tokens, memTotal: memGB * GB, budget });
    capTokens = Math.min(capTokens, tokens);
  }
  const seqTokens = input.promptLen + input.outputLen;
  const maxModelLen = input.maxModelLen;
  const fits = capTokens >= maxModelLen && capTokens > 0;
  const maxConc = Math.min(maxNumSeqs * par.pp, Math.floor(capTokens / seqTokens));
  T(tr, { group: "容量", label: "每副本最大并发（按 prompt+output 预留）", fn: "feasibility", formula: "min(max_num_seqs × PP, floor(KV tokens / (prompt + output)))", inputs: { KVtokens: capTokens, prompt: input.promptLen, output: input.outputLen, max_num_seqs: maxNumSeqs, PP: par.pp }, value: maxConc, unit: "seqs", kind: "formula" });
  let reason = null;
  if (capTokens <= 0) reason = "权重 + 激活 + 开销已超过可用显存";
  else if (capTokens < maxModelLen) reason = `KV 只够 ${capTokens} tokens，小于 max_model_len=${maxModelLen}`;
  return { fits, reason, stages, kvTokens: capTokens, maxConc: Math.max(0, maxConc) };
}

// ---------------------------------------------------------------------------
// 7. 通信代价
// ---------------------------------------------------------------------------
export function linkFor(gpu, groupSize, A) {
  // 组内 ≤ 单机 GPU 数 → NVLink（单向 = 双向/2）；否则瓶颈是跨机网卡（Gb/s → GB/s）
  if (groupSize <= gpu.gpusPerNode) return { name: "NVLink", GBps: gpu.nvlinkGBps / 2, alphaMs: A.allreduceLatencyMsNvlink };
  return { name: "IB/RoCE", GBps: gpu.scaleOutGbps / 8, alphaMs: A.allreduceLatencyMsIb };
}
export function allreduceMs(msgBytes, n, link) {
  if (n <= 1) return 0;
  return 2 * (n - 1) / n * msgBytes / (link.GBps * GB) * 1000 + link.alphaMs;
}
export function alltoallMs(msgBytes, n, link) {
  if (n <= 1) return 0;
  return (n - 1) / n * msgBytes / (link.GBps * GB) * 1000 + link.alphaMs;
}

// ---------------------------------------------------------------------------
// 8. Roofline：Decode 单步 & Prefill
// ---------------------------------------------------------------------------
function attnDim(model) {
  const c = model.config;
  if (model.attention === "mla") return c.num_attention_heads * (c.qk_nope_head_dim + c.qk_rope_head_dim);
  return c.num_attention_heads * (c.head_dim ?? c.hidden_size / c.num_attention_heads);
}
function computeFlops(gpu, prec) {
  // 仅当 GPU 有对应精度的张量核时才使用低精度 FLOPS（A100 上 FP8 = W8A16 Marlin，计算仍为 BF16）
  const p = prec.ffn === prec.attn ? prec.ffn : "bf16";
  if (p === "fp4" && gpu.supports.fp4Compute) return { flops: gpu.tflops.fp4, dtype: "fp4" };
  if (p === "fp8" && gpu.supports.fp8Compute) return { flops: gpu.tflops.fp8, dtype: "fp8" };
  return { flops: gpu.tflops.bf16, dtype: "bf16" };
}

export function decodeStep(input, par, feas, batch, tr = null) {
  const model = MODELS[input.modelId], gpu = GPUS[input.gpuId], A = input.assumptions, c = model.config;
  const h = c.hidden_size, ab = A.actBytes;
  const ctx = input.promptLen + input.outputLen / 2;
  const b = batch / par.pp; // 每个 PP stage 同时处理的 micro-batch
  const bw = T(tr, { group: "Decode · 访存", label: "单卡访存带宽 × 有效系数", fn: "decodeStep", formula: "memBandwidth × bandwidthEfficiency", inputs: { memBandwidth_GBps: gpu.memBandwidthGBps, bandwidthEfficiency: A.bandwidthEfficiency }, value: gpu.memBandwidthGBps * A.bandwidthEfficiency, unit: "GB/s", kind: "datasheet", src: gpu.sources.memBandwidthGBps.url + " — " + gpu.sources.memBandwidthGBps.quote });
  const cf = computeFlops(gpu, input.prec);
  const flops = T(tr, { group: "Decode · 算力", label: `单卡算力（${cf.dtype}）× MFU`, fn: "decodeStep", formula: "TFLOPS × 1e12 × mfuDecode", inputs: { TFLOPS: cf.flops, mfuDecode: A.mfuDecode }, value: cf.flops * A.mfuDecode, unit: "TFLOPS", kind: "datasheet", src: gpu.sources.tflops.url });
  const moe = model.family === "moe";
  const E = c.n_routed_experts ?? 1, k = c.num_experts_per_tok ?? 0;
  const tokensSeenByExperts = moe && par.epEnabled ? b * par.dp : b;
  const touched = moe ? 1 - Math.pow(1 - k / E, tokensSeenByExperts) : 0;
  if (moe) T(tr, { group: "Decode · 访存", label: "每步被激活的本地专家比例", fn: "decodeStep", formula: "1 − (1 − top_k/E)^(路由到本卡专家组的 token 数)", inputs: { top_k: k, E, tokens: +tokensSeenByExperts.toFixed(2) }, value: touched, unit: "", kind: "formula" });

  const tpLink = linkFor(gpu, par.tp, A);
  const epLink = linkFor(gpu, par.ep, A);
  const out = { stages: [], memMs: 0, compMs: 0, tpMs: 0, epMs: 0, ppMs: 0 };
  let total = 0;
  feas.stages.forEach((st, s) => {
    const w = st.weights;
    const kvRead = b * ctx * st.kvPerToken;
    const memBytes = w.nonExpertBytes + w.expertBytes * touched + kvRead;
    const memMs = memBytes / (bw * GB) * 1000;
    const nMoeLayers = moe ? w.layers - Math.max(0, Math.min(w.range[1], c.first_k_dense_replace) - w.range[0]) : 0;
    const oneExpertRank = moe ? 3 * h * c.moe_intermediate_size / (par.epEnabled ? 1 : par.tp) : 0; // 每卡上单个专家（单层）的参数
    const expertTokensRank = moe ? (par.epEnabled ? b * par.dp * k / par.ep : b * k) : 0;     // 每层路由到本卡的 token×专家 次数
    const expertFlops = moe ? 2 * nMoeLayers * oneExpertRank * expertTokensRank : 0;
    const attnFlops = 2 * w.layers * b * ctx * attnDim(model) / par.tp;
    const denseFlops = 2 * w.activeParams * b;
    const compMs = (denseFlops + expertFlops + attnFlops) / (flops * 1e12) * 1000;
    const nAllreduce = par.tp > 1 ? 2 * w.layers : 0;
    const tpMs = nAllreduce * allreduceMs(b * h * ab, par.tp, tpLink);
    const epMs = moe && par.epEnabled && par.ep > 1 ? 2 * nMoeLayers * alltoallMs(b * k * h * ab, par.ep, epLink) : 0;
    const stepMs = Math.max(memMs, compMs) + tpMs + epMs;
    if (tr && s === 0) {
      T(tr, { group: "Decode · 访存", label: "Stage 0 每步读取字节（每卡）", fn: "decodeStep", formula: "非专家权重 + 专家权重 × 激活比例 + micro_batch × 平均上下文 × 每token KV", inputs: { 非专家GB: +(w.nonExpertBytes / GB).toFixed(2), 专家GB: +(w.expertBytes / GB).toFixed(2), 激活比例: +touched.toFixed(3), micro_batch: +b.toFixed(2), 平均上下文: ctx, 每tokenKV: st.kvPerToken }, value: memBytes / GB, unit: "GB", kind: "formula" });
      T(tr, { group: "Decode · 访存", label: "Stage 0 访存时间", fn: "decodeStep", formula: "读取字节 / 有效带宽", inputs: { GB: +(memBytes / GB).toFixed(2), GBps: bw }, value: memMs, unit: "ms", kind: "formula" });
      T(tr, { group: "Decode · 算力", label: "Stage 0 计算时间", fn: "decodeStep", formula: "(2·激活参数·b + 2·专家参数·路由token + 2·层数·b·ctx·d_attn/TP) / (TFLOPS·MFU)", inputs: { 激活参数_rank: Math.round(w.activeParams), b: +b.toFixed(2), ctx, d_attn: attnDim(model), TP: par.tp }, value: compMs, unit: "ms", kind: "formula" });
      T(tr, { group: "Decode · 通信", label: `Stage 0 TP AllReduce（${tpLink.name}）`, fn: "allreduceMs", formula: "2·层数 × [2(n−1)/n · (b·h·act_bytes) / 链路带宽 + α]", inputs: { 次数: nAllreduce, n: par.tp, 消息bytes: Math.round(b * h * ab), 链路GBps: tpLink.GBps, α_ms: tpLink.alphaMs }, value: tpMs, unit: "ms", kind: "formula", src: tpLink.name === "NVLink" ? gpu.sources.nvlinkGBps.url : gpu.sources.scaleOutGbps.url });
      if (moe) T(tr, { group: "Decode · 通信", label: `Stage 0 EP All-to-All（${par.epEnabled ? epLink.name : "未开启"}）`, fn: "alltoallMs", formula: "2·MoE层数 × [(n−1)/n · (b·top_k·h·act_bytes) / 链路带宽 + α]", inputs: { MoE层数: nMoeLayers, n: par.epEnabled ? par.ep : 1, 链路GBps: epLink.GBps }, value: epMs, unit: "ms", kind: "formula" });
    }
    out.stages.push({ memMs, compMs, tpMs, epMs, stepMs });
    out.memMs += memMs; out.compMs += compMs; out.tpMs += tpMs; out.epMs += epMs;
    total += stepMs;
  });
  if (par.pp > 1) {
    const ppLink = { name: par.nodes > 1 ? "IB/RoCE" : "NVLink", GBps: par.nodes > 1 ? gpu.scaleOutGbps / 8 : gpu.nvlinkGBps / 2, alphaMs: par.nodes > 1 ? A.allreduceLatencyMsIb : A.allreduceLatencyMsNvlink };
    out.ppMs = (par.pp - 1) * (b * h * ab / (ppLink.GBps * GB) * 1000 + ppLink.alphaMs);
    T(tr, { group: "Decode · 通信", label: `PP 点对点发送（${ppLink.name}）`, fn: "decodeStep", formula: "(PP−1) × (b·h·act_bytes / 带宽 + α)", inputs: { PP: par.pp, b: +b.toFixed(2) }, value: out.ppMs, unit: "ms", kind: "formula" });
    total += out.ppMs;
  }
  out.tpotMs = T(tr, { group: "Decode · 结果", label: "TPOT（每输出 token 时延）", fn: "decodeStep", formula: "Σ_stage [max(访存, 计算) + TP通信 + EP通信] + PP发送", inputs: { batch, PP: par.pp }, value: total, unit: "ms", kind: "formula" });
  out.tokPerSecReplica = batch / (total / 1000);
  out.tokPerSec = T(tr, { group: "Decode · 结果", label: "系统输出吞吐", fn: "decodeStep", formula: "DP × batch / TPOT", inputs: { DP: par.dp, batch, TPOT_ms: +total.toFixed(2) }, value: par.dp * batch / (total / 1000), unit: "tok/s", kind: "formula" });
  out.tokPerSecPerGpu = out.tokPerSec / par.gpus;
  const mem = out.memMs, comp = out.compMs;
  out.bottleneck = Math.max(mem, comp) === mem ? "访存带宽" : "算力";
  if (out.tpMs + out.epMs + out.ppMs > Math.max(mem, comp)) out.bottleneck = "通信";
  return out;
}

export function prefill(input, par, feas, tr = null) {
  const model = MODELS[input.modelId], gpu = GPUS[input.gpuId], A = input.assumptions, c = model.config;
  const P = input.promptLen, h = c.hidden_size, ab = A.actBytes;
  const cf = computeFlops(gpu, input.prec);
  const flops = cf.flops * A.mfuPrefill * 1e12;
  const moe = model.family === "moe", E = c.n_routed_experts ?? 1, k = c.num_experts_per_tok ?? 0;
  const tpLink = linkFor(gpu, par.tp, A), epLink = linkFor(gpu, par.ep, A);
  let total = 0, comp = 0, comm = 0;
  feas.stages.forEach((st, s) => {
    const w = st.weights;
    const nMoeL = moe ? w.layers - Math.max(0, Math.min(w.range[1], c.first_k_dense_replace) - w.range[0]) : 0;
    const oneExpertRank = moe ? 3 * h * c.moe_intermediate_size / (par.epEnabled ? 1 : par.tp) : 0;
    const expertTokensRank = moe ? (par.epEnabled ? P * k / par.ep : P * k) : 0;
    const f = 2 * w.activeParams * P + 2 * nMoeL * oneExpertRank * expertTokensRank + w.layers * P * P * attnDim(model) / par.tp;
    const cm = f / flops * 1000;
    const tpm = par.tp > 1 ? 2 * w.layers * allreduceMs(P * h * ab, par.tp, tpLink) : 0;
    const epm = moe && par.epEnabled && par.ep > 1 ? 2 * nMoeL * alltoallMs(P * k * h * ab, par.ep, epLink) : 0;
    comp += cm; comm += tpm + epm; total += cm + tpm + epm;
    if (tr && s === 0) T(tr, { group: "Prefill", label: "Stage 0 计算量（每卡）", fn: "prefill", formula: "2·激活参数·P + 2·专家参数·路由token + 层数·P²·d_attn/TP", inputs: { P, 激活参数_rank: Math.round(w.activeParams), d_attn: attnDim(model), TP: par.tp }, value: f, unit: "FLOPs", kind: "formula" });
  });
  if (par.pp > 1) {
    const GBps = par.nodes > 1 ? gpu.scaleOutGbps / 8 : gpu.nvlinkGBps / 2;
    const ppMs = (par.pp - 1) * (P * h * ab / (GBps * GB) * 1000);
    comm += ppMs; total += ppMs;
  }
  T(tr, { group: "Prefill", label: `算力（${cf.dtype}）× MFU`, fn: "prefill", formula: "TFLOPS × mfuPrefill", inputs: { TFLOPS: cf.flops, mfuPrefill: A.mfuPrefill }, value: cf.flops * A.mfuPrefill, unit: "TFLOPS", kind: "datasheet", src: gpu.sources.tflops.url });
  const ttft = T(tr, { group: "Prefill", label: "TTFT（单请求、无排队）", fn: "prefill", formula: "Σ_stage [FLOPs / 有效算力 + TP/EP 通信] + PP 发送", inputs: { 计算ms: +comp.toFixed(1), 通信ms: +comm.toFixed(1) }, value: total, unit: "ms", kind: "formula" });
  return { ttftMs: ttft, compMs: comp, commMs: comm };
}

// ---------------------------------------------------------------------------
// 9. 并行配置枚举
// ---------------------------------------------------------------------------
const POW2 = [1, 2, 4, 8, 16, 32];
export function enumerateParallel(input) {
  const model = MODELS[input.modelId], gpu = GPUS[input.gpuId], c = model.config, G = gpu.gpusPerNode;
  const out = [];
  for (let nodes = 1; nodes <= input.maxNodes; nodes++) {
    const gpus = nodes * G;
    for (const tp of POW2) {
      if (tp > gpus) continue;
      if (c.num_attention_heads % tp) continue;
      if (model.attention === "gqa") { const kv = c.num_key_value_heads; if (kv % tp && tp % kv) continue; }
      for (const pp of [1, 2, 3, 4, 6, 8]) {
        if (tp * pp > gpus || gpus % (tp * pp)) continue;
        const dp = gpus / (tp * pp);
        if (dp > 1 && nodes > 1 && (tp * pp > G || G % (tp * pp))) continue; // 多机 DP 要求单个副本在一台机内
        if (pp > 1 && nodes > 1 && tp > G) continue;
        if (tp > G && pp === 1 && nodes > 2) continue;
        const epChoices = model.family === "moe" ? [false, true] : [false];
        for (const epEnabled of epChoices) {
          const ep = epEnabled ? tp * dp : 1;
          if (epEnabled && (ep < 2 || c.n_routed_experts % ep)) continue;
          if (epEnabled && pp > 1) continue; // 简化：EP 场景不与 PP 叠加
          out.push({ nodes, gpus, tp, pp, dp, ep, epEnabled });
        }
      }
    }
  }
  return out;
}

export function parLabel(p) {
  return `${p.nodes}机${p.gpus}卡 · TP${p.tp}${p.pp > 1 ? " PP" + p.pp : ""}${p.dp > 1 ? " DP" + p.dp : ""}${p.epEnabled ? " EP" + p.ep : ""}`;
}

// 在 TPOT SLA 约束下找最大 batch（每副本）
export function bestBatch(input, par, feas) {
  const cap = feas.maxConc;
  if (cap < 1) return null;
  const sla = input.tpotSlaMs || Infinity;
  const want = input.targetConcurrency ? Math.ceil(input.targetConcurrency / par.dp) : cap;
  let lo = 1, hi = Math.min(cap, input.objective === "latency" ? want : cap), best = null;
  const d1 = decodeStep(input, par, feas, 1);
  if (d1.tpotMs > sla) return { batch: 1, perf: d1, slaMet: false };
  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);
    const d = decodeStep(input, par, feas, mid);
    if (d.tpotMs <= sla) { best = { batch: mid, perf: d, slaMet: true }; lo = mid + 1; } else hi = mid - 1;
  }
  return best;
}

export function evaluate(input, par, tr = null) {
  const feas = feasibility(input, par, tr);
  const res = { par, label: parLabel(par), feas };
  if (!feas.fits) return res;
  const bb = bestBatch(input, par, feas);
  const opBatch = input.objective === "latency"
    ? Math.max(1, Math.min(feas.maxConc, Math.ceil((input.targetConcurrency || 1) / par.dp)))
    : bb?.batch ?? 1;
  res.batch = opBatch;
  res.decode = decodeStep(input, par, feas, opBatch, tr);
  res.decode1 = decodeStep(input, par, feas, 1);
  res.prefill = prefill(input, par, feas, tr);
  res.capacity = feas.maxConc * par.dp;
  res.meetsConcurrency = !input.targetConcurrency || res.capacity >= input.targetConcurrency;
  res.meetsSla = !input.tpotSlaMs || res.decode.tpotMs <= input.tpotSlaMs;
  return res;
}

export function searchConfigs(input) {
  const all = enumerateParallel(input).map((p) => evaluate(input, p));
  const ok = all.filter((r) => r.feas.fits);
  const key = {
    throughput: (r) => -r.decode.tokPerSecPerGpu,
    latency: (r) => r.decode.tpotMs,
    cost: (r) => r.par.gpus * 1e6 - r.decode.tokPerSecPerGpu,
  }[input.objective];
  // 主指标相差 <5% 视为持平，持平时优先更少 GPU，再优先更少节点
  const cmp = (a, b) => {
    const ka = key(a), kb = key(b);
    if (input.objective !== "cost" && Math.abs(ka - kb) / Math.max(Math.abs(ka), Math.abs(kb), 1e-9) < 0.05)
      return a.par.gpus - b.par.gpus || a.par.nodes - b.par.nodes || ka - kb;
    return ka - kb;
  };
  const good = ok.filter((r) => r.meetsConcurrency && r.meetsSla).sort(cmp);
  const rest = ok.filter((r) => !(r.meetsConcurrency && r.meetsSla)).sort(cmp);
  // Pareto：TPOT 越小越好、每卡吞吐越大越好
  const pareto = new Set();
  for (const r of ok) {
    if (!ok.some((o) => o !== r && o.decode.tpotMs <= r.decode.tpotMs && o.decode.tokPerSecPerGpu >= r.decode.tokPerSecPerGpu && (o.decode.tpotMs < r.decode.tpotMs || o.decode.tokPerSecPerGpu > r.decode.tokPerSecPerGpu))) pareto.add(r);
  }
  for (const r of ok) r.pareto = pareto.has(r);
  return { ranked: [...good, ...rest], infeasible: all.filter((r) => !r.feas.fits), all };
}

export function minNodesSingleReplica(input) {
  // 最少需要几台机才能装下一个副本（用于提示「单机放不下」）
  const gpu = GPUS[input.gpuId];
  for (let n = 1; n <= 8; n++) {
    const cands = enumerateParallel({ ...input, maxNodes: n }).filter((p) => p.nodes === n);
    if (cands.some((p) => feasibility(input, p).fits)) return n;
  }
  return null;
}

export function batchCurve(input, par) {
  const feas = feasibility(input, par);
  if (!feas.fits) return [];
  const pts = [];
  for (let b = 1; b <= feas.maxConc; b = b < 4 ? b + 1 : Math.ceil(b * 1.35)) {
    const d = decodeStep(input, par, feas, b);
    pts.push({ batch: b, tpotMs: d.tpotMs, tokPerSecPerGpu: d.tokPerSecPerGpu, memMs: d.memMs, compMs: d.compMs, commMs: d.tpMs + d.epMs + d.ppMs });
  }
  const d = decodeStep(input, par, feas, feas.maxConc);
  if (!pts.length || pts[pts.length - 1].batch !== feas.maxConc) pts.push({ batch: feas.maxConc, tpotMs: d.tpotMs, tokPerSecPerGpu: d.tokPerSecPerGpu, memMs: d.memMs, compMs: d.compMs, commMs: d.tpMs + d.epMs + d.ppMs });
  return pts;
}

// ---------------------------------------------------------------------------
// 10. vllm serve 命令
// ---------------------------------------------------------------------------
export function vllmCommands(input, res) {
  const model = MODELS[input.modelId], p = res.par, gpu = GPUS[input.gpuId];
  const notes = [];
  const base = [`vllm serve ${model.hfRepo}`];
  const common = [];
  if (p.tp > 1) common.push(`--tensor-parallel-size ${p.tp}`);
  if (p.pp > 1) common.push(`--pipeline-parallel-size ${p.pp}`);
  if (p.epEnabled) common.push(`--enable-expert-parallel`);
  common.push(`--max-model-len ${input.maxModelLen}`);
  common.push(`--max-num-seqs ${Math.max(1, Math.ceil(res.batch / p.pp))}`);
  common.push(`--max-num-batched-tokens ${input.assumptions.maxNumBatchedTokens}`);
  common.push(`--gpu-memory-utilization ${input.assumptions.gpuMemoryUtilization}`);
  if (input.kvDtype === "fp8") common.push(`--kv-cache-dtype fp8`);
  const precs = new Set([input.prec.attn, input.prec.ffn]);
  if (model.nativeDtype === "bf16" && precs.has("fp8") && precs.size === 1) { common.push(`--quantization fp8`); notes.push("在线 FP8 量化（逐张量 scale），显存与 block FP8 基本一致。"); }
  else if (![...precs].every((x) => x === model.nativeDtype)) notes.push(`所选精度与官方权重（${model.nativeDtype.toUpperCase()}）不同：需要先准备对应精度的预量化权重，并把 ${model.hfRepo} 替换为该权重路径。`);
  if (gpu.id === "A100" && precs.has("fp8")) notes.push("A100 无 FP8 张量核：vLLM 以 W8A16（FP8 Marlin）运行，只省显存不加速计算。");
  const cmds = [];
  const fmt = (arr) => arr.join(" \\\n    ");
  if (p.nodes === 1) {
    if (p.dp > 1) common.unshift(`--data-parallel-size ${p.dp}`);
    cmds.push({ title: "单机", cmd: fmt([...base, ...common]) });
  } else if (p.dp > 1) {
    const local = p.dp / p.nodes;
    for (let n = 0; n < p.nodes; n++) {
      const a = [...base, ...(n > 0 ? ["--headless"] : []), `--data-parallel-size ${p.dp}`, `--data-parallel-size-local ${local}`, ...(n > 0 ? [`--data-parallel-start-rank ${n * local}`] : []), `--data-parallel-address $HEAD_IP`, `--data-parallel-rpc-port 13345`, ...common];
      cmds.push({ title: `节点 ${n}${n === 0 ? "（Head，提供 API）" : ""}`, cmd: fmt(a) });
    }
    notes.push("多机数据并行写法见 vLLM 文档：" + VLLM_DOCS.dataParallel);
  } else {
    for (let n = 0; n < p.nodes; n++) {
      const a = [...base, ...common, `--nnodes ${p.nodes}`, `--node-rank ${n}`, `--master-addr $HEAD_IP`, ...(n > 0 ? ["--headless"] : [])];
      cmds.push({ title: `节点 ${n}${n === 0 ? "（Head，提供 API）" : ""}`, cmd: fmt(a) });
    }
    notes.push("多机 TP/PP（MultiProcessing 后端）写法见 vLLM 文档：" + VLLM_DOCS.multiNode + "；也可用 Ray：--distributed-executor-backend ray。");
    if (p.tp > gpu.gpusPerNode) notes.push("⚠ TP 跨机：每层 AllReduce 走网卡，通常远慢于 TP=单机卡数 + PP。");
  }
  return { cmds, notes };
}

export function defaultInput(overrides = {}) {
  return {
    modelId: "deepseek-v3", gpuId: "A100",
    prec: { attn: "fp8", ffn: "fp8" }, kvDtype: "auto",
    promptLen: 2048, outputLen: 512, maxModelLen: 8192,
    targetConcurrency: 64, tpotSlaMs: 50, maxNumSeqs: 1024,
    objective: "throughput", maxNodes: 4,
    assumptions: { ...DEFAULT_ASSUMPTIONS },
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// 11. 用实测 TPOT 反推访存带宽有效系数（校准）
//     思路：TPOT = Σ max(访存, 计算) + 通信。若访存是瓶颈，
//     访存时间 ∝ 1/η，因此 η_new = η_old × 预测访存 / (实测 − 通信)
// ---------------------------------------------------------------------------
export function calibrateBandwidth(input, par, batch, measuredTpotMs) {
  const feas = feasibility(input, par);
  if (!feas.fits) return null;
  const d = decodeStep(input, par, feas, batch);
  const comm = d.tpMs + d.epMs + d.ppMs;
  const target = measuredTpotMs - comm;
  if (target <= 0 || d.memMs < d.compMs) return { ok: false, reason: d.memMs < d.compMs ? "该 batch 下计算是瓶颈，带宽系数无法由 TPOT 反推；请用更小 batch 的实测值" : "实测值小于通信估算，请检查输入", predicted: d.tpotMs };
  const eta = input.assumptions.bandwidthEfficiency * d.memMs / target;
  return { ok: true, eta, predicted: d.tpotMs, formula: "η_new = η_old × 预测访存ms / (实测TPOT − 通信ms)", inputs: { η_old: input.assumptions.bandwidthEfficiency, 预测访存ms: d.memMs, 实测TPOT: measuredTpotMs, 通信ms: comm } };
}

// 显存分解（给图表用，单位 GB，每卡）
export function memoryBreakdown(input, par) {
  const feas = feasibility(input, par);
  const util = input.assumptions.gpuMemoryUtilization;
  return feas.stages.map((st) => {
    const b = st.weights.breakdown;
    const kv = Math.max(0, st.kvBudget);
    return {
      stage: st.stage, layers: st.weights.range, fits: st.kvBudget > 0,
      parts: [
        { key: "attn", label: "注意力权重", gb: b.attn / GB },
        { key: "ffn", label: model_ffn_label(input), gb: (b.ffn + b.experts) / GB },
        { key: "other", label: "Embedding/Norm/Router", gb: b.other / GB },
        { key: "act", label: "激活峰值", gb: st.act / GB },
        { key: "overhead", label: "CUDA/NCCL 开销", gb: st.overhead / GB },
        { key: "kv", label: "KV Cache", gb: kv / GB },
        { key: "reserve", label: `预留 (1−${util})`, gb: st.memTotal * (1 - util) / GB },
      ],
      overGB: st.kvBudget < 0 ? -st.kvBudget / GB : 0,
      totalGB: st.memTotal / GB, kvTokens: st.kvTokens,
    };
  });
}
function model_ffn_label(input) { return MODELS[input.modelId].family === "moe" ? "专家/FFN 权重" : "FFN 权重"; }

// 有效算力（供 UI 展示 Prefill/Decode 对照用；与 decodeStep/prefill 内部逻辑一致）
export function effectiveFlops(gpuId, prec) {
  const gpu = GPUS[gpuId];
  const p = prec.ffn === prec.attn ? prec.ffn : "bf16";
  if (p === "fp4" && gpu.supports.fp4Compute) return { peak: gpu.tflops.fp4, dtype: "fp4" };
  if (p === "fp8" && gpu.supports.fp8Compute) return { peak: gpu.tflops.fp8, dtype: "fp8" };
  return { peak: gpu.tflops.bf16, dtype: "bf16" };
}
