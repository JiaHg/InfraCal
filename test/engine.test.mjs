// node --test test/
import test from "node:test";
import assert from "node:assert/strict";
import { MODELS } from "../data.js";
import * as E from "../engine.js";

const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) / b <= tol, `${msg}: got ${a}, expected ≈${b} (±${tol * 100}%)`);

test("参数量与官方公布值对照", () => {
  const l = E.paramSummary(MODELS["llama-3.3-70b"]);
  near(l.total / 1e9, 70.6, 0.01, "Llama-3.3-70B total");
  const d = E.paramSummary(MODELS["deepseek-v3"]);
  near(d.total / 1e9, 671, 0.01, "DeepSeek-V3 total");
  near(d.active / 1e9, 37, 0.03, "DeepSeek-V3 active");
  near(d.mtp / 1e9, 14, 0.05, "DeepSeek-V3 MTP");
});

test("KV 每 token 字节（手推公式对照）", () => {
  const par = { tp: 1, pp: 1 };
  // Llama: 2 × 80 × 8 × 128 × 2 = 327,680
  assert.equal(E.kvBytesPerToken(MODELS["llama-3.3-70b"], par, "auto", 80), 327680);
  // TP=16 > n_kv=8 → 每卡 1 个 KV 头
  assert.equal(E.kvBytesPerToken(MODELS["llama-3.3-70b"], { tp: 16 }, "auto", 80), 2 * 80 * 1 * 128 * 2);
  // MLA: 61 × (512 + 64) × 2 = 70,272，且不随 TP 变化
  assert.equal(E.kvBytesPerToken(MODELS["deepseek-v3"], par, "auto", 61), 70272);
  assert.equal(E.kvBytesPerToken(MODELS["deepseek-v3"], { tp: 8 }, "auto", 61), 70272);
});

test("精度字节数", () => {
  assert.equal(E.bytesPerWeight("bf16"), 2);
  near(E.bytesPerWeight("fp8"), 1 + 4 / 16384, 1e-12, "fp8 block");
  assert.equal(E.bytesPerWeight("fp4"), 0.5625);
});

test("TP 切分后各卡权重之和 ≈ 全量权重（去掉复制部分）", () => {
  const m = MODELS["llama-3.3-70b"];
  const w1 = E.stageWeights(m, { tp: 1, pp: 1, ep: 1 }, { attn: "bf16", ffn: "bf16" }, 0).total;
  near(w1 / 1e9, 70.55 * 2, 0.01, "Llama BF16 TP1 weights GB");
  const w8 = E.stageWeights(m, { tp: 8, pp: 1, ep: 1 }, { attn: "bf16", ffn: "bf16" }, 0).total;
  assert.ok(w8 * 8 >= w1 && w8 * 8 < w1 * 1.01);
});

test("PP 层切分覆盖全部层", () => {
  for (const [L, pp] of [[61, 2], [61, 4], [80, 3], [80, 8]]) {
    const r = E.ppLayerRanges(L, pp);
    assert.equal(r[0][0], 0); assert.equal(r.at(-1)[1], L);
    for (let i = 1; i < r.length; i++) assert.equal(r[i][0], r[i - 1][1]);
  }
});

test("DeepSeek-V3 FP8 在单台 8×A100 放不下，需要多机；8×B200 单机可放", () => {
  const a = E.defaultInput({ modelId: "deepseek-v3", gpuId: "A100" });
  assert.ok(E.minNodesSingleReplica(a) >= 2);
  const b = E.defaultInput({ modelId: "deepseek-v3", gpuId: "B200" });
  assert.equal(E.minNodesSingleReplica(b), 1);
});

test("搜索结果：可行、有序、有命令", () => {
  const inp = E.defaultInput({ modelId: "llama-3.3-70b", gpuId: "A100", prec: { attn: "bf16", ffn: "bf16" }, maxNodes: 2 });
  const r = E.searchConfigs(inp);
  assert.ok(r.ranked.length > 0);
  const top = r.ranked[0];
  assert.ok(top.decode.tpotMs > 0 && top.decode.tokPerSecPerGpu > 0);
  const cmd = E.vllmCommands(inp, top);
  assert.ok(cmd.cmds[0].cmd.startsWith("vllm serve"));
});

test("Decode 访存时间单调：batch 越大 TPOT 越大、单卡吞吐越高", () => {
  const inp = E.defaultInput({ modelId: "llama-3.3-70b", gpuId: "B200", prec: { attn: "bf16", ffn: "bf16" } });
  const par = { nodes: 1, gpus: 8, tp: 8, pp: 1, dp: 1, ep: 1, epEnabled: false };
  const f = E.feasibility(inp, par);
  const d1 = E.decodeStep(inp, par, f, 1), d32 = E.decodeStep(inp, par, f, 32);
  assert.ok(d32.tpotMs > d1.tpotMs);
  assert.ok(d32.tokPerSecPerGpu > d1.tokPerSecPerGpu);
});

test("Trace 记录每一步且带函数名", () => {
  const inp = E.defaultInput();
  const tr = new E.Trace();
  E.evaluate(inp, { nodes: 2, gpus: 16, tp: 8, pp: 2, dp: 1, ep: 1, epEnabled: false }, tr);
  assert.ok(tr.steps.length > 15);
  for (const s of tr.steps) { assert.ok(s.fn in E, `fn ${s.fn} exported`); assert.ok(Number.isFinite(s.value) || typeof s.value === "number"); }
});
