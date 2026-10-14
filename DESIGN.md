# Infra_cal 设计文档（v1）

网页：https://claude.ai/artifact/HhTfZMZxqfXYp7hCGNgwzz

## 范围
- 模型：Llama-3.3-70B（Dense/GQA，BF16）、DeepSeek-V3（MoE/MLA，官方 FP8 block 128×128）
- GPU：B200 180GB、A100 80GB（8 卡/机）
- 框架：vLLM；输出 `vllm serve` 单机/多机命令（MP 后端 --nnodes/--node-rank；DP 用 --data-parallel-*；MoE 可 --enable-expert-parallel）

## 文件
- data.js：模型 config 原文 + GPU 规格（每个字段带来源 URL 与原文引用）+ 可调假设
- engine.js：纯函数引擎，所有函数接受 Trace，记录 {group,label,formula,inputs,value,unit,kind,src,fn}
- app.js：UI（控件、推荐方案、拓扑、命令、Pareto 散点、配置表、显存分解、Batch 扫描、计算轨迹）
- test/engine.test.mjs：`node --test test/engine.test.mjs`，9 项（参数量对照 70.6B/671B/37B/14B MTP、KV 公式、PP 切分、单机可行性等）

## 三层模型
1. 可行性：组件级参数分解（切分规则 tp/rep/kv/vocab/expert）→ 每卡权重（混合精度：注意力 vs FFN/专家）→ 激活峰值（经验公式）+ 开销 → KV 预算 → 并发容量。MLA 潜向量 KV 在 TP 各卡复制。
2. Roofline：Decode 每 stage = max(访存, 计算) + TP AllReduce + EP All-to-All，再加 PP 发送；MoE 访存按 1−(1−k/E)^tokens 计本地专家激活比例。Prefill = FLOPs/(TFLOPS×MFU) + 通信。A100 上 FP8 = W8A16 Marlin，算力按 BF16。
3. 搜索：枚举 节点×TP×PP×DP×EP，约束（heads 可整除、多机 DP 副本在单机内、EP 不叠加 PP），目标：吞吐优先（SLA 下二分最大 batch）/ 时延优先 / 最少 GPU；主指标差 <5% 时优先少卡。

## 数据来源
- DGX B200 页：64 TB/s HBM3e（÷8=8TB/s）、NVLink 14.4TB/s（÷8=1.8TB/s）、ConnectX-7 400Gb/s
- A100 datasheet：2,039GB/s、NVLink 600GB/s；DGX A100：ConnectX-6 200Gb/s
- TFLOPS：stas00/ml-engineering（A100 bf16 312；B200 bf16 2250 / fp8 4500 / fp4 9000）
- vLLM FP8 在 Ampere 以 W8A16 Marlin 运行（v0.8.1 文档）

## 已知局限 / 下一步
- 未建模：排队调度、prefix caching、MTP 投机解码、chunked prefill 干扰、PD 分离
- 官方带宽为理论值，结果偏乐观 → 页面提供用实测 TPOT 反推 η 的校准
- NVFP4 在 A100 上未建模；EP+PP 组合未枚举
- 扩展：在 data.js 追加模型 config / GPU 条目即可（需补 MODEL_DEFAULTS 精度默认值）
