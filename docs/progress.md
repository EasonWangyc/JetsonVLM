# 项目进展记录

更新时间：2026-07-26

## 1. 当前结论

仓库已经建立从泊车样本到研究报告的代码链路：

```text
ParkingCase -> RiskRuntime -> InferenceRecord -> StudyReport
```

领域对象、冻结工作负载、Transformers Adapter、TensorRT Edge-LLM HTTP Adapter、
质量与性能汇总、命令入口以及外部流程包装器已经实现，并通过无硬件单元测试。

Jetson 环境已经完成连接和基础检查，指定 revision 的
`Qwen/Qwen3-VL-2B-Instruct` 已下载到板端并完成缓存完整性校验。

当前尚未执行真实单图推理、冻结测试集研究、LoRA 训练或 TensorRT Edge-LLM engine
构建。因此，“代码链路已建立”和“模型文件已就绪”不等同于“板端基线已经完成”。

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
| 外部流程 | LoRA、合并、导出和 engine 构建的 dry-run/execute 包装器 | 静态入口与校验逻辑 |
| Jetson 准备 | SSH、系统、CUDA、Python/CUDA 依赖、功耗模式、内存和磁盘检查 | 板端命令输出 |
| 模型准备 | 固定模型 commit、下载 12 个文件并校验权重哈希 | Hugging Face 缓存与校验结果 |

## 3. Jetson 环境快照

| 项目 | 当前值 |
| --- | --- |
| 设备 | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super |
| L4T | R36.4.7 |
| CUDA Toolkit | 12.6 |
| 功耗模式 | `15W`，mode id `0` |
| Python 环境 | `/home/ubuntu/project/llm-on-device/.venv`，Python 3.12.12 |
| PyTorch | `2.9.1+cu126`，`torch.cuda.is_available() == True` |
| Transformers | `4.57.6` |
| Accelerate | `1.12.0` |
| Pillow | `12.1.0` |
| Hugging Face Hub CLI | `0.36.0` |
| 系统内存 | 7.4 GiB |
| 当前可用内存 | 5.7 GiB |
| Swap | 3.7 GiB zram，当前实际使用约 1 MiB |
| NVMe | 233 GiB，总剩余约 181 GiB |

`free -h` 中较小的 `free` 值主要来自 Linux 将空闲内存用于 page cache；判断是否
存在内存压力应优先查看 `available` 和实际 swap 使用量。当前快照没有显示明显内存
压力。zram 设备处于启用状态，但不代表系统正在大量换页。

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

## 6. 下一阶段

1. 准备一张许可明确的泊车图片和对应最小 case manifest。
2. 在 Jetson 上执行 Qwen3-VL Transformers FP16 单图 smoke test。
3. 同步采集启动状态、端到端耗时、CUDA 峰值内存、系统内存和失败事实。
4. 根据首次结果决定是否调整输入尺寸或生成 token 上限；不改变冻结 JSON schema。
5. 扩展冻结测试集并运行 Jetson Transformers FP16 `StudyReport`。
6. 之后再进入服务器正确性参考、LoRA 和 TensorRT Edge-LLM engine 阶段。
