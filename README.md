# Infra_cal · 大模型推理部署计算器

给定**模型 + GPU + 负载**，计算显存能不能放下、需要几台机器、怎么切并行最划算，并直接给出 `vllm serve` 多机启动命令。页面上的每个数字都能点开查看公式、代入值、官方来源和计算它的源码。

**在线访问：** https://jiahg.github.io/InfraCal/

## 功能

| 模块 | 做什么 |
|---|---|
| 可行性 | 按组件拆参数量（GQA / MLA / Dense FFN / MoE 专家），算出每卡权重、激活峰值、开销、KV 预算和并发容量 |
| Roofline | Decode 取「访存 / 计算」较大者，再加 TP AllReduce、EP All-to-All、PP 发送；Prefill 按算力 × MFU 加通信 |
| 方案搜索 | 枚举 节点 × TP × PP × DP × EP，按吞吐 / 时延 / 省卡排序，给出 Pareto 前沿 |
| 部署 | 生成单机或多机 `vllm serve` 命令（`--nnodes/--node-rank` 或 `--data-parallel-*`、`--enable-expert-parallel`） |
| 可追溯 | 每一步记录 `{公式, 代入值, 结果, 来源, 函数名}`，可导出 JSON，也能直接查看函数源码 |
| 校准 | 输入实测 TPOT，反推访存带宽有效系数 η |

首批支持 **Llama-3.3-70B**（Dense / GQA）和 **DeepSeek-V3**（MoE / MLA，FP8），GPU 支持 **B200 180GB** 和 **A100 80GB**。

## 结构

```
index.html            页面（Emerald Editorial 风格）
app.js                UI：收集输入 → 调用引擎 → 渲染，不做数值推导
engine.js             计算引擎：纯函数，Node 和浏览器都能跑
data.js               模型 config 原文 + GPU 官方规格（每个字段带来源）+ 可调假设
test/engine.test.mjs  单元测试（参数量对照官方值、KV 公式、PP 切分、单机可行性等）
```

## 本地运行

纯静态站点，没有构建步骤。ES module 需要通过 HTTP 访问：

```bash
python3 -m http.server 8080    # 然后打开 http://localhost:8080
node --test test/              # 跑单元测试（Node ≥ 18）
```

## 扩展

- **加模型**：在 `data.js` 的 `MODELS` 里加一条（把 HF `config.json` 原样粘进 `config` 字段），再在 `app.js` 的 `MODEL_DEFAULTS` 里设默认精度。
- **加 GPU**：在 `data.js` 的 `GPUS` 里加一条，每个规格字段都附上来源链接。

## 数据来源

- 模型：HuggingFace `config.json`（[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)、[Llama-3.3-70B](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/blob/main/config.json)）
- GPU：[DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/)、[A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf)、[DGX A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-dgx-a100-datasheet.pdf)、[TFLOPS 对照表](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md)
- vLLM：[多机部署](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/)、[数据并行](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/)、[Ampere 上的 FP8](https://docs.vllm.ai/en/v0.8.1/features/quantization/fp8.html)

## 局限

这是 roofline **估算**，不是基准测试。没有建模：请求排队与调度、prefix caching、投机解码（含 MTP）、chunked prefill 对 decode 的干扰、PD 分离。带宽和算力用的是官方理论值，实际 TPOT 通常更高，拿到实测数据后用页面上的「校准」反推 η 最可靠。

## License

MIT
