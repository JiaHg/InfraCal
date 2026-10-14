// ============================================================================
// Infra_cal · 数据层
// 所有数字均来自官方原文（链接见 source 字段）。标记为 assumption 的是经验假设，
// 在页面「高级参数」中可调，且计算轨迹里会显式标注。
// 扩展方式：新增一个模型 = 在 MODELS 里追加一条（config 原样粘贴）；
//          新增一种 GPU = 在 GPUS 里追加一条。
// ============================================================================

export const MODELS = {
  "llama-3.3-70b": {
    id: "llama-3.3-70b",
    name: "Llama-3.3-70B-Instruct",
    family: "dense",          // dense | moe
    attention: "gqa",         // gqa | mla
    hfRepo: "meta-llama/Llama-3.3-70B-Instruct",
    source: {
      label: "HF config.json（meta-llama 官方仓库为 gated，内容取自无门槛镜像 unsloth/Llama-3.3-70B-Instruct，架构字段一致）",
      url: "https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/blob/main/config.json",
      official: "https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct",
    },
    publishedParamsB: 70.6,   // 官方模型卡标称 70B（用于单元测试对照，允许误差）
    nativeDtype: "bf16",
    config: {
      architectures: ["LlamaForCausalLM"],
      head_dim: 128,
      hidden_size: 8192,
      intermediate_size: 28672,
      max_position_embeddings: 131072,
      num_attention_heads: 64,
      num_hidden_layers: 80,
      num_key_value_heads: 8,
      tie_word_embeddings: false,
      torch_dtype: "bfloat16",
      vocab_size: 128256,
    },
  },
  "deepseek-v3": {
    id: "deepseek-v3",
    name: "DeepSeek-V3",
    family: "moe",
    attention: "mla",
    hfRepo: "deepseek-ai/DeepSeek-V3",
    source: {
      label: "HF config.json（deepseek-ai/DeepSeek-V3）",
      url: "https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json",
      official: "https://huggingface.co/deepseek-ai/DeepSeek-V3",
    },
    publishedParamsB: 671,       // 官方 README：671B 总参数（不含 14B MTP 模块）
    publishedActiveB: 37,        // 官方 README：每 token 激活 37B
    publishedMtpB: 14,
    publishedSourceUrl: "https://huggingface.co/deepseek-ai/DeepSeek-V3#1-introduction",
    nativeDtype: "fp8",
    config: {
      architectures: ["DeepseekV3ForCausalLM"],
      first_k_dense_replace: 3,
      hidden_size: 7168,
      intermediate_size: 18432,
      kv_lora_rank: 512,
      max_position_embeddings: 163840,
      moe_intermediate_size: 2048,
      moe_layer_freq: 1,
      n_group: 8,
      n_routed_experts: 256,
      n_shared_experts: 1,
      num_attention_heads: 128,
      num_experts_per_tok: 8,
      num_hidden_layers: 61,
      num_key_value_heads: 128,
      num_nextn_predict_layers: 1,
      q_lora_rank: 1536,
      qk_nope_head_dim: 128,
      qk_rope_head_dim: 64,
      quantization_config: { activation_scheme: "dynamic", fmt: "e4m3", quant_method: "fp8", weight_block_size: [128, 128] },
      tie_word_embeddings: false,
      topk_group: 4,
      torch_dtype: "bfloat16",
      v_head_dim: 128,
      vocab_size: 129280,
    },
  },
};

const SRC = {
  a100: "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf",
  dgxA100: "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-dgx-a100-datasheet.pdf",
  dgxB200: "https://www.nvidia.com/en-us/data-center/dgx-b200/",
  b200Lenovo: "https://lenovopress.lenovo.com/lp2226-thinksystem-nvidia-b200-180gb-1000w-gpu",
  tflops: "https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md#tflops-comparison-table",
  vllmFp8Ampere: "https://docs.vllm.ai/en/v0.8.1/features/quantization/fp8.html",
};

export const GPUS = {
  B200: {
    id: "B200",
    name: "NVIDIA B200 180GB",
    memGB: 180,
    memBandwidthGBps: 8000,      // DGX B200: 64 TB/s HBM3e 总带宽 ÷ 8 GPU
    nvlinkGBps: 1800,            // DGX B200: 14.4 TB/s NVLink 聚合带宽 ÷ 8 GPU（双向）
    gpusPerNode: 8,
    scaleOutGbps: 400,           // 每 GPU 一张 ConnectX-7 400Gb/s
    tflops: { bf16: 2250, fp8: 4500, fp4: 9000 },
    computeCapability: 10.0,
    supports: { fp8Compute: true, fp4Compute: true },
    sources: {
      memGB: { url: SRC.dgxB200, quote: "1,440 GB total GPU memory (8 GPU)" },
      memBandwidthGBps: { url: SRC.dgxB200, quote: "64 TB/s HBM3e bandwidth（8 GPU 合计，÷8 = 8 TB/s）" },
      nvlinkGBps: { url: SRC.dgxB200, quote: "14.4 TB/s aggregate NVLink bandwidth（÷8 = 1.8 TB/s，双向）" },
      scaleOutGbps: { url: SRC.dgxB200, quote: "8x single-port ConnectX-7 VPI, up to 400 Gb/s InfiniBand/Ethernet" },
      tflops: { url: SRC.tflops, quote: "B200: bf16 2250 / fp8 4500 / fp4 9000 TFLOPS（理论峰值）" },
    },
  },
  A100: {
    id: "A100",
    name: "NVIDIA A100 80GB",
    memGB: 80,
    memBandwidthGBps: 2039,
    nvlinkGBps: 600,             // NVLink 3 双向
    gpusPerNode: 8,
    scaleOutGbps: 200,           // DGX A100: 8x ConnectX-6 HDR 200Gb/s
    tflops: { bf16: 312, fp8: null, fp4: null },
    computeCapability: 8.0,
    supports: { fp8Compute: false, fp4Compute: false },
    sources: {
      memGB: { url: SRC.a100, quote: "80GB HBM2e" },
      memBandwidthGBps: { url: SRC.a100, quote: "Memory Bandwidth 2,039GB/s" },
      nvlinkGBps: { url: SRC.a100, quote: "NVLink 600GB/s" },
      scaleOutGbps: { url: SRC.dgxA100, quote: "8x Single-Port Mellanox ConnectX-6 VPI 200Gb/s HDR InfiniBand" },
      tflops: { url: SRC.tflops, quote: "A100: bf16 312 TFLOPS（理论峰值），无 FP8/FP4 张量核" },
      fp8: { url: SRC.vllmFp8Ampere, quote: "FP8 models will run on compute capability > 8.0 (Ampere) as weight-only W8A16, utilizing FP8 Marlin." },
    },
  },
};

// 精度定义：bytesPerParam 含量化 scale 开销（计算过程见 engine.bytesPerWeight）
export const PRECISIONS = {
  bf16: { id: "bf16", label: "BF16", bits: 16, groupSize: null, scaleBits: 0 },
  fp8: { id: "fp8", label: "FP8 (block 128×128)", bits: 8, groupSize: 128 * 128, scaleBits: 32,
    source: "https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json (weight_block_size [128,128], 每块一个 float32 scale)" },
  fp4: { id: "fp4", label: "NVFP4", bits: 4, groupSize: 16, scaleBits: 8,
    source: "https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ (每 16 个值一个 FP8 scale，另有每张量 FP32 scale 可忽略)" },
};

export const KV_DTYPES = {
  auto: { id: "auto", label: "BF16（auto）", bytes: 2 },
  fp8: { id: "fp8", label: "FP8", bytes: 1 },
};

export const VLLM_DOCS = {
  multiNode: "https://docs.vllm.ai/en/latest/serving/parallelism_scaling/",
  dataParallel: "https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/",
  expertParallel: "https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/",
};

// 经验假设（全部可在 UI 调整；轨迹中标注为「假设」）
export const DEFAULT_ASSUMPTIONS = {
  gpuMemoryUtilization: 0.9,   // vLLM --gpu-memory-utilization 默认值
  nonTorchOverheadGB: 2.0,     // CUDA context + NCCL buffer 等非 PyTorch 显存
  cudaGraphGB: 1.0,            // CUDA Graph 捕获占用
  maxNumBatchedTokens: 8192,   // 激活峰值 profiling 用的 token 数（--max-num-batched-tokens）
  mfuPrefill: 0.4,             // Prefill 的算力利用率
  mfuDecode: 0.4,              // Decode 计算侧利用率
  bandwidthEfficiency: 1.0,    // 访存带宽有效系数（1 = 纯 roofline 上界；可用实测值校准）
  allreduceLatencyMsNvlink: 0.03, // 单次 NVLink 集合通信固定延迟
  allreduceLatencyMsIb: 0.3,   // 单次跨机集合通信固定延迟
  actBytes: 2,                 // 激活精度（BF16）
};
