# 项目进展记录

更新时间：2026-08-12

## 1. 当前结论

仓库已经建立从泊车样本到研究报告的代码链路：

```text
ParkingCase -> RiskRuntime -> InferenceRecord -> StudyReport
```

领域对象、冻结工作负载、Transformers Adapter、TensorRT Edge-LLM HTTP Adapter、
质量与性能汇总、命令入口以及外部流程包装器已经实现，并通过无硬件单元测试。

Jetson 已升级到 L4T R36.5.0 / JetPack 6.2.2。固定 revision 的模型与服务器 ONNX
归档均已在板端复算哈希；固定 commit 的 TensorRT Edge-LLM 已编译，动态库依赖、
Python binding 和 builder/runtime 可执行文件均通过检查。

启用临时 8 GiB 磁盘 swap 后，视觉与 LLM 两个 FP16 engine 均已在目标 Jetson 构建
成功并形成独立 `succeeded` flow record。运行时通过
`setWeightStreamingBudgetV2(0)` 在 8 GB 设备上同时加载两套 engine，HTTP 健康检查、
真实单图和冻结 20 样本 study 均已执行。

修复多模态 system content 契约后，使用补丁后的 runtime 和缩小到
`maxInputLen=768`、`maxKVCacheCapacity=1024` 的 FP16 engine 重新运行冻结 20 样本。
20/20 样本均完成后端推理并通过严格 schema，失败汇总为空。端到端 p50/p90/p99 为
50.75/69.08/75.08 秒；542 条 `tegrastats` 显示平均板端输入功耗 10.05 W、GPU 峰值
温度 65.03°C。风险等级准确率为 35%，事件 micro-F1 为 0.359，说明部署与格式链路
已经闭环，但领域质量仍需要 LoRA/数据改进。

从项目开始至今的完整命令、结果与证据见
[`execution-report.md`](execution-report.md)。

## 2. 已完成工作

| 范围 | 已完成内容 | 当前证据 |
| --- | --- | --- |
| 风险输出 | `ParkingAssessment` 严格 JSON 解析、枚举和字段校验 | 单元测试 |
| 数据入口 | manifest、annotation、图片引用、来源组和数据划分校验 | 单元测试与 fixtures |
| 工作负载 | prompt、生成参数、输入尺寸和 schema 的冻结配置及 SHA-256 身份 | 单元测试 |
| Transformers | Qwen3-VL 懒加载、图片预处理、生成、输出解码、耗时和显存记录 | Adapter 代码与 mock 测试 |
| Edge-LLM | OpenAI-compatible HTTP 请求和响应解析 | Adapter 代码与 mock 测试 |
| 运行记录 | 成功结果、JSON 失败、输入失败、超时和 OOM 等失败事实 | 单元测试 |
| 研究汇总 | 质量指标、性能分位数、资源数据、失败汇总和环境快照 | 单元测试 |
| 应用入口 | 单图分析、配置化研究和 runtime factory | 单元测试 |
| 外部流程 | LoRA、合并、导出和可拆分 engine 构建的 dry-run/execute 包装器 | 静态入口、服务器 ONNX record、Jetson LLM/visual engine record |
| Jetson 准备 | SSH、系统、CUDA、Python/CUDA 依赖、功耗模式、内存和磁盘检查 | 板端命令输出 |
| 模型准备 | 固定模型 commit、下载 12 个文件并校验权重哈希 | Hugging Face 缓存与校验结果 |
| FP16 engine | 独立构建 LLM 与视觉 engine，保留哈希和 flow record | 两个 `succeeded` flow record |
| FP16 runtime | 插件加载、权重流式预算、双 engine、CUDA graph 与 HTTP 服务 | server log 与 `/health` 200 |
| 单图/20 样本 | 真实图片推理、20/20 严格 JSON 成功与同机遥测 | StudyReport、runtime summary、542 条 tegrastats |

## 3. Jetson 环境快照

| 项目 | 当前值 |
| --- | --- |
| 设备 | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super |
| L4T / JetPack | R36.5.0 / 6.2.2 |
| CUDA Toolkit | 12.6 |
| 功耗模式 | `15W`，mode id `0` |
| Python 环境 | `/home/ubuntu/JetsonVLM/.venv-jetson`，Python 3.10.12 |
| PyTorch | `2.9.1`，CUDA 可用，arch list 为 `['sm_87']` |
| Transformers | `4.57.6` |
| Accelerate | `1.12.0` |
| Pillow | `12.1.0` |
| Hugging Face Hub CLI | `0.36.0` |
| 系统内存 | 7.4 GiB |
| 当前可用内存 | 空闲时约 6.8 GiB |
| Swap | 3.7 GiB zram + 已启用的 8 GiB 临时磁盘 swap；未写入 `fstab` |
| NVMe | 233 GiB，当前剩余约 161 GiB |

`free -h` 中较小的 `free` 值主要来自 Linux 将空闲内存用于 page cache；判断是否
存在内存压力应优先查看 `available` 和实际 swap 使用量。空闲快照没有明显压力；
LLM engine 构建时则有内核 OOM 与 TensorRT 分配失败的直接证据。

## 4. 模型版本与缓存

- 模型：`Qwen/Qwen3-VL-2B-Instruct`
- 固定 commit：`89644892e4d85e24eaac8bacfd4f463576704203`
- snapshot：
  `/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203`
- 仓库缓存占用：约 4.0 GiB
- 校验结果：12/12 文件存在，字节数为 `4266648961/4266648961`，
  `.incomplete` 文件为 0，`model.safetensors` SHA-256 与 LFS blob 哈希一致
- 原有 `Qwen3-0.6B` 缓存仍保留，约 1.5 GiB

同一个 commit 已写入服务器 Transformers、Jetson Transformers FP16、TensorRT
Edge-LLM FP16 和 LoRA 训练示例配置，避免不同实验阶段静默使用不同权重。

## 5. 已执行的主要命令

以下命令按用途整理。`<JETSON_HOST>` 表示本次已验证可连接的 Jetson SSH 地址。

### 5.1 连接与系统检查

```bash
ssh <JETSON_HOST>
cat /proc/device-tree/model
cat /etc/nv_tegra_release
/usr/local/cuda/bin/nvcc --version
nvpmodel -q
free -h
swapon --show --bytes
df -h /
```

### 5.2 Python/CUDA 环境检查

```bash
/home/ubuntu/project/llm-on-device/.venv/bin/python -V
/home/ubuntu/project/llm-on-device/.venv/bin/hf version
```

另通过该 Python 环境导入 `torch`、`transformers`、`accelerate` 和 `Pillow`，
记录版本并检查 `torch.cuda.is_available()`。

### 5.3 解析并下载固定模型 revision

先通过 `huggingface_hub.HfApi.model_info()` 读取官方仓库 SHA，再按 SHA 下载：

```bash
/home/ubuntu/project/llm-on-device/.venv/bin/hf download \
  Qwen/Qwen3-VL-2B-Instruct \
  --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --cache-dir /home/ubuntu/.cache/huggingface/hub
```

下载后通过 `HfApi.model_info(..., files_metadata=True)` 对比远端文件名和字节数，
递归检查 `.incomplete` 文件，并对 `model.safetensors` 计算 SHA-256。

### 5.4 仓库验证

本次 Windows 主机未在 `cmd.exe` 的 `PATH` 中发现 `uv`，因此使用等价的 Python
入口并显式设置 `PYTHONPATH`：

```bat
set PYTHONPATH=src&python -m unittest discover -s tests
git diff --check
```

结果为 27 个测试全部通过，`git diff --check` 通过。

### 5.5 已确认的工具版本兼容性

板端 `huggingface_hub==0.36.0` 不提供以下较新 CLI 能力：

```text
hf models info
hf download --dry-run
hf cache verify
```

因此本次使用 `HfApi.model_info()` 解析 commit 和文件元数据，并使用本地文件对比与
权重 SHA-256 完成等价校验，没有为此升级或修改板端 Python 环境。

## 6. Jetson 单图 smoke test 记录

### 2026-07-26：CLI 参数失败

- 命令在模型加载前退出。
- 原因：当时的单图入口不支持 `--dtype float16`。
- 结果文件为空，stderr 明确记录 `unrecognized arguments`。
- 后续 commit `5e2bdfd` 已补齐 `dtype`、`device_map` 和
  `attn_implementation` 的 CLI 传递与测试。

### 2026-07-27：Transformers FP16 run1

- 代码 revision：`5e2bdfd`
- 模型 revision：`89644892e4d85e24eaac8bacfd4f463576704203`
- 配置：`dtype=float16`、`device_map=auto`、`sdpa`
- 结果：失败，退出码 2，未进入 GPU 计算
- `InferenceRecord`：`runtime_error`
- 端到端失败时间：约 15.25 秒
- stderr：NvMap 约 1.0 GiB 分配返回 error 12，随后 PyTorch CUDA allocator
  触发 NVML 内部断言
- `tegrastats`：RAM 峰值约 4.0/7.6 GiB，`GR3D_FREQ` 保持 0%

进一步 CUDA 自检确认当前环境不是 Jetson 可执行构建：

```text
torch=2.9.1+cu126
device=Orin
capability=(8, 7)
arch_list=['sm_80', 'sm_90']
64 MiB CUDA tensor: no kernel image is available for execution on the device
```

当前 torch 来自 PyTorch CUDA 12.6 通用
`cp312-manylinux_2_28_aarch64` wheel。它能发现 Orin，但不包含 Orin 所需的 `sm_87`
kernel，因此不能作为 Jetson Transformers 基线环境。后续应建立独立 Python 3.10
Jetson runtime，使用 JP6/CUDA 12.6 的 Jetson 专用 wheel，或使用 NVIDIA iGPU
PyTorch 容器；不得覆盖现有 Python 3.12 venv。

commit `731fa2f` 增加 CUDA 架构预检后，run2 在约 7.09 秒内返回：

```text
category=dependency_unavailable
installed torch 2.9.1+cu126 does not include CUDA kernels for sm_87;
supported architectures: sm_80, sm_90
```

run2 的 stderr 为空，说明该问题现在由 `InferenceRecord` 稳定记录，不再依赖底层
NvMap/NVML 错误文本归因。

### 2026-07-28：Jetson 专用环境与 run3-run5

- 建立 `/home/ubuntu/JetsonVLM/.venv-jetson`，Python 3.10.12。
- 安装 Jetson AI Lab PyTorch 2.9.1 和 torchvision 0.24.1。
- 安装 cuDSS 0.4.0.2.post1 并显式配置其动态库路径。
- `torch.cuda.get_arch_list()` 返回 `['sm_87']`，小 tensor CUDA kernel 成功。
- 新环境运行 29 个无硬件测试，全部通过。
- run3 因缺少 torchvision 失败，随后补齐依赖。
- run4 在模型加载期间出现 NvMap error 12 和 PyTorch NVML allocator assertion。
- run5 关闭 CUDA caching allocator 后仍出现 NvMap error 12，并被记录为
  `out_of_memory`。
- run5 RAM 峰值约 4005/7620 MiB，swap 为 0，最小 lfb 为 2×4 MiB，
  `GR3D_FREQ` 为 0%；尚未进入视觉编码或生成。

## 7. 当前实验结果与下一阶段

### 7.1 TensorRT Edge-LLM FP16 结果

| 指标 | 实测结果 |
| --- | --- |
| 单图稳定性验收 | 同一图片连续 3/3 严格 JSON 成功，平均 76.19 s |
| 20 样本后端完成率 | 20/20 |
| 严格 JSON 有效率 | 20/20（100%） |
| 失败类型 | 无 |
| 风险等级准确率 / 事件 micro-F1 | 35% / 0.359 |
| 不安全建议率 | 0% |
| 端到端 p50/p90/p99 | 50.75/69.08/75.08 s |
| 聚合端到端输出速率 | 1.48 token/s |
| RAM / swap 峰值 | 7418/1904 MB |
| GPU 利用率均值 | 97.39% |
| 板端输入功耗均值 | 10.05 W |
| GPU 峰值温度 | 65.03°C |

原始证据保存在本机忽略目录 `reports/jetson-runtime-20260812/`。派生摘要由
`scripts/summarize_jetson_study.py` 从原始 StudyReport 与 tegrastats 生成，既不修改
原始记录，也不把 JSON 失败计为业务成功。

### 7.2 LoRA 与 INT4 状态

当前 20 个样本全部属于冻结 `test` 集，不能直接拿来训练。LoRA 示例配置仍引用尚未
建立的 `parking_risk_v1.jsonl` 训练/验证数据，训练与合并命令仍为占位符，因此本轮
没有执行 LoRA 或生成 adapter。下一步应先从独立来源组建立 train/validation 数据，
重点监督数组字段、六类领域事件和安全建议，再在 GPU 服务器运行 LoRA 并回到同一
冻结 test 集复测。

INT4 尚无校准集、量化配置或 engine。FP16 运行已经证明 0-byte 权重驻留预算能在
8 GB Jetson 上运行，但代价是 p50 约 50.75 秒；INT4 的首要目标是降低权重与 scratch
压力、提高 GPU 驻留比例和 token 吞吐，而不是把尚未执行的计划写成完成结果。

同机加速比还需要补齐 Jetson Transformers FP16 的同一 20 样本报告；早期 OOM 记录
保留为明确实验结果，但不能用服务器 Transformers 延迟替代板端基线。
