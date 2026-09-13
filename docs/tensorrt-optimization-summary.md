# TensorRT Edge-LLM 优化阶段速览

本文只记录近期 TensorRT Edge-LLM 性能优化的固定条件、已核验结果和候选结论。完整命令、
实验过程、provenance 字段和原始证据索引见 [完整实验记录](tensorrt-optimization.md)。

## 当前状态

当前正式参考为 TensorRT Edge-LLM INT4 level 0 engine；level 1 为已完成性能复验的候选。
本轮优化变量集中在 builder tactic、CUDA Graph、INT4 GEMV/GEMM tile、`lm_head` GEMM 和
低层 benchmark，不把量化、LoRA、数据标注或 workload 作为变量。

## 固定实验身份

| 项目 | 固定值 |
|---|---|
| 设备 | Jetson Orin Nano Super，SM87，15W_MODE_0 |
| 软件 | JetPack R36.5，CUDA 12.6.68，TensorRT 10.3.0.30 |
| Edge-LLM | v0.9.1，commit `7f061f21f0a581ba234a1e233c9315b89d8e47d6` |
| 模型 | Qwen3-VL-2B-Instruct，revision `89644892e4d85e24eaac8bacfd4f463576704203` |
| 精度 | INT4 AWQ LLM backbone，FP16 `lm_head` |
| profile | batch=1，`i768/k1024`，KV capacity=1024 |
| 构建 | workspace=1024 MiB，CUDA Graph enabled，weight streaming disabled |
| 插件 SHA-256 | `9437996d36b659092e7d4da244b43da6feb7c69e79a5654a8df53e9f5c2ac7eb` |
| level 0 engine SHA-256 | `33466f3f1149801bf496737fbe1e69634b8018ece9a0c5bccd8ff83af64afab4` |
| level 1 engine SHA-256 | `207e109fcca28ac29ae0348d8dd517e61a6291704a8fea08c53644574f28d345` |

## 最终已核验性能

### 基线与完整请求

| 阶段 | Runtime / 配置 | Prefill | TTFT | Decode / generation | E2E p50 / p90 / p99 |
|---|---|---:|---:|---:|---:|
| 基线 | Jetson Transformers FP16 | 486.32 ms | 716.69 ms | 7,897.63 ms/request；9.54 tok/s | 9.569 / 14.439 / 28.155 s |
| 阶段 1 | TensorRT Edge-LLM INT4，level 0 | 531.61 ms | 872.66 ms<sup>1</sup> | 124.02 ms/token；8.06 tok/s | 10.523 / 11.387 / 12.723 s |
| 阶段 2 | TensorRT Edge-LLM INT4，level 1 | 430.97 ms | 702.50 ms<sup>1</sup> | 27.25 ms/token；36.70 tok/s | 2.826 / 2.953 / 3.315 s |

基线为 `1×20` 请求，Edge-LLM 两档为 `3×20` 请求。不同 runtime 的 decode profile 不是
严格 A/B；TTFT 是请求发送到 SSE 首个非空 `delta.content` 的客户端时间，包含视觉编码、
调度、首 token decode、网络和流式输出，因此不能用 prefill 代替。E2E 包含预处理、视觉编码、
prefill、decode、HTTP 和序列化。

### Edge-LLM 低层严格 A/B

低层 workload 固定 `inputLen=768`、`pastKVLen=768`、OSL=1、warm-up=10、每次 20 次、
seed=0，并对 level 0/1 各做 3 次重复。

| 阶段 | 构建变量 | Prefill（20 次均值） | Decode Graph（20 次均值） | Decode 吞吐 | 相对 level 0 |
|---|---|---:|---:|---:|---:|
| level 0 | `builderOptimizationLevel=0` | 526.8711 ms | 126.6875 ms/token | 7.9 tok/s | — |
| level 1 | `builderOptimizationLevel=1` | 426.7465 ms | 27.4129 ms/token | 36.5 tok/s | prefill `-19.039%`；decode `4.6215×` |
| level 1 soak | level 1，CUDA Graph，1000 steps | — | 27.3821 ms/token | 36.5202 tok/s | 0 次 runtime failure |

level 1 相对 level 0 的 decode latency 降低 `78.3618%`。1000-step soak 进程退出码为 0、
Graph capture 成功；同时保留独立 metadata loader 的 TensorRT runtime destructor warning。

## Nsight Systems 与 Engine Inspector

### 主要 tactic 差异

Engine Inspector 显示 level 0/1 的层数和 INT4 W4A16 主体保持一致，主要结构性差异集中在
最终 `lm_head` GEMM。该矩阵为：

```text
[1, 1, 2048] × [1, 2048, 151936] -> [1, 1, 151936]
```

| 热点 | level 0 | level 1 | 变化 |
|---|---:|---:|---:|
| 最终 FP16 `lm_head` GEMM（单次 trace） | 111.4517 ms | 10.9825 ms | `-90.14%` |
| INT4 W4A16 主体（196 instances） | 262.3125 ms | 262.3898 ms | `+0.03%` |
| FMHA head-dim 64（24 instances） | 62.0573 ms | 57.4192 ms | `-7.48%` |

level 1 non-Graph decode trace 的 GPU kernel 时间分布为：

| Kernel | 总时间 | Trace share |
|---|---:|---:|
| `gemv_kernel<2,1,256,128>` | 141.1500 ms | 52.3% |
| `trt_ampere_h16816gemm_128x64_ldg8_tn_v1`（FP16 `lm_head`） | 92.2431 ms | 34.2% |
| `kernel_mha` | 18.6402 ms | 6.9% |

当前瓶颈优先级是大词表 `lm_head` tactic、INT4 decode GEMV 以及 launch/Graph 行为，
不能将 level 1 的收益笼统描述为“所有算子都融合或加速”。Nsight 的 CUDA API 总时间包含
初始化和 warm-up，不能直接当作单步 decode 时间。

## 算子与矩阵候选结论

| 候选 | 测试口径 | 结果 | 结论 |
|---|---|---|---|
| builder level `0 -> 1` | 整体 engine tactic | decode `4.6215×`，prefill `-19.039%` | 已核验的主要收益阶段 |
| INT4 GEMV block/tile `128/192/320/512` | batch=1 decode | `128` 约 `-0.15%`；`192/320/512` 变慢约 `1.32%/7.15%/4.31%` | 未形成稳定收益 |
| `CTA_N=256` | INT4 W4A16 prefill GEMM | `+17.24%` | 否决 |
| `CTA_M=128` | INT4 W4A16 prefill GEMM | `+16.67%` | 否决 |
| `__ldg` read-only cache hint | INT4 GEMV decode | no-Graph `-0.17%`，Graph `+0.04%` | 否决 |
| `__restrict__` pointer hint | INT4 GEMV decode | no-Graph `-0.08%`，Graph `+0.56%` | 否决 |
| `NPerBlock=4` + CUDA stream 修复 | INT4 GEMV decode | 仅完成候选 Graph 局部运行 | control/A-B 和完整 VLM 未完成，不纳入结果 |

上述候选都保持 engine、权重布局、scale 布局和 workload 不变，并在隔离 plugin stack 上构建；
正式 plugin 和正式 engine 没有被这些候选覆盖。

## 当前技术边界

- 当前路径使用 Edge-LLM 已有的 INT4 W4A16、attention/FMHA plugin、TensorRT tactic 选择和 CUDA Graph。
- 当前项目增加的是构建配置、provenance、benchmark、Nsight 汇总和隔离候选 patch；没有将自研
  CUDA kernel 或新 TensorRT plugin 宣称为正式生产路径。
- 投机采样、NVFP4、FP8、paged KV cache 和 FP16 weight streaming 尚未形成本轮可比较的性能结果。
- builder level `2/3` 在板端构建阶段因约 622 MB CUDA allocation 失败，尚未生成可比较 engine；
  不对其推理性能作推断。

## 证据入口

- [阶段 1 报告](../reports/jetson-tensorrt-stage1/phase1_summary.md)
- [level 0/1 prefill 复验](../reports/jetson-tensorrt-revalidation/int4_level0_level1_prefill768_decode20_20260909.json)
- [level 0/1 decode 重复性](../reports/jetson-tensorrt-revalidation/int4_level0_level1_decode20_repeats_20260909.json)
- [level 1 1000-step soak](../reports/jetson-tensorrt-revalidation/int4_level1_decode1000_soak_20260909.json)
- [Nsight 对比摘要](../reports/jetson-tensorrt-revalidation/int4_opt0_vs_opt1_nsight_summary.json)
- [level 1 热点摘要](../reports/jetson-tensorrt-revalidation/int4_level1_formal_nograph_nsys_20260909.json)
- [GEMV/GEMM 候选 A/B 目录](../reports/jetson-tensorrt-revalidation)
