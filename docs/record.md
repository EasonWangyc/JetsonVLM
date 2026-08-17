# ParkSight-VLM 完整项目学习与执行记录

更新时间：2026-08-12
当前已推送基线：`6eb178b feat: complete Jetson Edge-LLM FP16 acceptance`
当前本地增量：Prompt 契约与 Edge-LLM 内存分配顺序已修复；Jetson FP16 单图连续 3 次严格 JSON 验收通过

## 0. 如何阅读这份记录

这份文件按项目实际开发顺序，记录从仓库重构、Jetson 环境准备、模型下载、
Transformers FP16 基线、数据集 pilot，到 TensorRT Edge-LLM 的 ONNX 导出、板端编译、
engine 构建和真实推理验收的全过程。

为了避免把“写了代码”误写成“已经跑通”，本文区分三种状态：

- **已实现**：仓库中已有代码、配置或命令入口，并由单元测试或静态检查覆盖。
- **已实测**：保存了终端结果、结构化报告、日志、产物哈希或板端遥测。
- **未完成**：只有设计、示例配置或占位入口，没有相应运行证据。

更偏证据审计的逐项报告见 [`execution-report.md`](execution-report.md)，当前状态摘要见
[`status.md`](status.md)，PS2.0 pilot 说明见 [`ps20-pilot.md`](ps20-pilot.md)，
Edge-LLM 操作手册见 [`edgellm-deployment.md`](edgellm-deployment.md)。本文件侧重把这些
内容串成一条便于复习的完整学习主线。

## 1. 项目最终目标与当前结论

### 1.1 项目要解决什么问题

项目输入是一张泊车场景图片；视频输入可以先抽帧，再把每一帧作为单图输入。模型输出
不是自由文本，而是经过严格校验的结构化 JSON，用于描述：

- 风险等级 `risk_level`；
- 风险事件 `events`；
- 图像证据 `evidence`；
- 驾驶建议 `driver_advice`；
- schema 版本 `schema_version`。

风险事件枚举固定为：

```text
vru_near_maneuver_path
vehicle_near_maneuver_path
fixed_obstacle_near_path
narrow_passage
visibility_occlusion
parking_space_conflict
```

固定枚举的价值是：数据标注、模型输出、运行时解析、质量评测和错误归因使用同一套数据
类型，不依赖自然语言的模糊解释。

### 1.2 软件主链路

```text
ParkingCase
  -> RiskRuntime
  -> InferenceRecord
  -> StudyReport
```

| 对象 / Module | 作用 |
| --- | --- |
| `ParkingCase` | 图片、样本 ID、来源组、划分和人工标注的组合 |
| `ParkingAssessment` | 对模型输出执行严格 JSON、字段、类型和枚举校验 |
| `RiskRuntime` | 屏蔽 Transformers 与 Edge-LLM HTTP 后端差异 |
| `InferenceRecord` | 保存一次推理的输入身份、运行时身份、原始输出、解析结果、耗时和失败事实 |
| `StudyRunner` | 在冻结 casebook 上批量运行并汇总质量、性能、环境和失败 |
| `StudyReport` | 可追溯的实验结果，不把失败样本从统计中静默删除 |

核心代码目录：

```text
src/parksight_vlm/
  assessment/       严格输出契约
  casebook/         manifest、annotation、来源组和划分
  inference/        Transformers 与 Edge-LLM Adapter
  studies/          运行记录、指标、报告和 Jetson 证据汇总
  app/              单图和批量 study 命令入口
```

### 1.3 三类实验角色

| 位置 | Runtime | 用途 |
| --- | --- | --- |
| GPU 服务器 | Transformers | 正确性参考、误差分析、LoRA/SFT 训练 |
| Jetson | Transformers FP16 | 板端框架基线 |
| Jetson | TensorRT Edge-LLM FP16 / INT4 | 最终部署和推理优化 |

GPU 服务器不能替代 Jetson 性能基线。性能对比必须保证模型 revision、workload、数据集、
生成参数和 Jetson 功耗模式一致。

### 1.4 截至当前真正完成的链路

已经在目标 Jetson 上实测完成：

```text
固定 Hugging Face 模型 revision
  -> GPU 服务器导出 LLM/visual ONNX
  -> ONNX 归档与 SHA-256 校验
  -> Jetson 编译固定 commit 的 TensorRT Edge-LLM
  -> Jetson 构建 visual.engine 和 llm.engine
  -> Edge-LLM 双 engine HTTP 服务
  -> 单图与 20 样本推理
  -> InferenceRecord / StudyReport / tegrastats 派生摘要
```

同时，Jetson Transformers FP16 已在相同 20 样本、相同 workload、相同模型 revision
和 15W 模式下形成成功报告。因此目前不是“只有整体框架”，而是已经有两套板端 Runtime
的真实结果。

当前最重要的事实是：

- Transformers FP16：20/20 严格 JSON 有效，端到端 p50 为 `9.385 s`；
- Edge-LLM FP16：20/20 后端请求完成，但严格 JSON 有效为 0/20，端到端 p50 为
  `41.758 s`；
- 当前 Edge-LLM 使用 0-byte GPU weight-streaming budget 才能在 8 GB 统一内存上同时
  加载双 engine，结果反而比 Transformers 慢；
- 所以项目已经证明“部署链路可运行”，但尚未证明“TensorRT 已加速”或“业务质量达标”。

## 2. 仓库演进与关键提交

| 日期 | commit | 主要内容 |
| --- | --- | --- |
| 2026-07-01 | `14089ed` | 初始 Jetson Visual Memory Agent scaffold |
| 2026-07-05 | `e0eccf6` | embodied inference deployment workflow |
| 2026-07-17 | `1b59764` | 早期仓库重构 |
| 2026-07-21 | `10346a0` | 重新定义仓库结构和目的 |
| 2026-07-26 | `bbb5715` | 建立 ParkSight-VLM 领域对象、Runtime 与 Study 主链路 |
| 2026-07-27 | `5e2bdfd` | 单图入口传递 `dtype`、`device_map`、attention 配置 |
| 2026-07-27 | `731fa2f` | 在模型加载前检查 CUDA 架构是否包含 `sm_87` |
| 2026-07-27 | `f362a43` | 保存 Jetson FP16 早期失败证据 |
| 2026-07-29 | `b031005` | 增加完整执行报告 |
| 2026-08-01 | `1e464cc` | 源码注释中文化 |
| 2026-08-02 | `61376c1` | 修复 Qwen3-VL 多模态 system message 格式 |
| 2026-08-02 | `6643f7f` | 下载 PS2.0、建立 20 样本标注并完成 Transformers pilot |
| 2026-08-04 | `c6ca2d6` | 增加 TensorRT Edge-LLM FP16 导出与部署 flow |
| 2026-08-10 | `6eb178b` | 完成 Jetson Edge-LLM FP16 engine 和推理验收 |

复查历史：

```bash
git log --oneline --decorate --all
git show --stat bbb5715
git show --stat 6643f7f
git show --stat c6ca2d6
git show --stat 6eb178b
```

早期 CLIP、SmolVLM、视觉记忆和 episode replay 设计属于仓库历史，不能作为当前
Qwen3-VL + TensorRT Edge-LLM 已完成证据。

## 3. 当前仓库结构与各文件的职责

```text
configs/
  workloads/parking_risk_v1.json              冻结 prompt、schema 和生成参数
  studies/                                    不同 Runtime 的实验身份
  flows/                                      导出与 engine 构建流程
  training/lora_v1.example.json               LoRA 示例，尚未可执行
data/
  manifests/ps20_pilot_v1.jsonl               20 个样本的图片引用、来源组和 test 划分
  annotations/ps20_pilot_v1.jsonl             20 个样本的领域标注
  raw/ps2.0/                                  原始数据，不作为可提交核心产物
patches/tensorrt-edge-llm/                    固定上游 commit 上的兼容性补丁
scripts/
  export_model.py                             ONNX 导出 flow 包装器
  build_engine.py                             TensorRT engine 构建 flow 包装器
  serve_edgellm.py                            只加载预构建 engine 的 HTTP 服务入口
  summarize_jetson_study.py                   StudyReport + tegrastats 派生摘要
  train_lora.py / merge_lora.py               训练与合并流程入口，当前未执行
reports/                                     原始运行记录和派生证据
tests/                                       无硬件单元测试
```

冻结 workload 的最终身份：

```text
parking_risk_v1@sha256:8350ace4574f8aa154319f7136ef831003d4dcc074ef20b74c1b419d69a2a493
input_size=448x448
max_new_tokens=256
do_sample=false
```

## 4. PC、GPU 服务器与 Jetson 的分工

### 4.1 Windows PC

PC 是主要代码编辑、文档、Git 和证据归档位置。项目代码修改后可以：

- commit/push 后让 Jetson pull；
- 或只把本轮改动文件通过 `scp` 发送到 Jetson；
- 板端不作为个人文档和 Git 历史的唯一存储位置。

Windows 本机可以进行无硬件测试，也可以做部分 CPU 侧模型下载和 ONNX 相关操作；但
是否能顺利导出仍受 Linux 专用依赖、内存、磁盘和上游工具实现限制。服务器导出不是
因为 ONNX 数学上只能在 GPU 生成，而是因为 Linux + NVIDIA 环境更接近 Edge-LLM
官方路径，且后续 LoRA 必须依赖大显存 GPU。

### 4.2 GPU 服务器

已租用 AutoDL RTX 4090 D 实例执行固定模型下载和 ONNX 导出。服务器后续主要用于：

- Transformers 正确性参考；
- 建立训练/验证数据后执行 LoRA/SFT；
- adapter 合并和合并模型复测；
- 必要时重新导出合并模型；
- 离线质量评测和误差分析。

TensorRT engine 不能直接在 4090 D 上构建后拿到 Jetson 使用，因为 engine 与 GPU
架构、TensorRT 版本、插件和构建参数相关。因此最终 engine 必须在 Jetson 目标环境
构建。

### 4.3 Jetson

Jetson 承担：

- Transformers FP16 同机基线；
- TensorRT Edge-LLM 源码编译；
- FP16/INT4 engine 构建；
- 最终图片推理与 Study；
- RAM、swap、GPU 利用率、功耗和温度采集。

Docker 在本项目中主要用于隔离复杂 NVIDIA 运行环境、复用已配好的 CUDA/PyTorch
镜像；它不是业务链路的必要中间层。当前最终 Jetson 验收采用仓库内
`.venv-jetson` 和板端系统 CUDA/TensorRT，未依赖 Docker 容器持续运行。

## 5. Jetson 环境检查、升级与资源理解

### 5.1 连接与基本检查

```bash
ssh ubuntu@192.168.137.187
cd /home/ubuntu/JetsonVLM

cat /proc/device-tree/model
cat /etc/nv_tegra_release
/usr/local/cuda/bin/nvcc --version
apt-cache policy nvidia-jetpack nvidia-l4t-core
nvpmodel -q
free -h
swapon --show --bytes
df -h /
```

最初实测：

| 项目 | 初始结果 |
| --- | --- |
| 设备 | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super |
| L4T | R36.4.7 |
| JetPack | 6.2.1 |
| CUDA | 12.6 |
| 功耗模式 | 15W，mode id 0 |
| RAM | 7.4 GiB |
| Swap | 3.7 GiB zram |
| NVMe | 233 GiB，初期可用约 181 GiB |

升级并重启后：

| 项目 | 当前实测结果 |
| --- | --- |
| L4T / JetPack | R36.5.0 / 6.2.2 |
| CUDA Toolkit | 12.6 |
| TensorRT | 10.3 |
| Python | 3.10.12，仓库专用 `.venv-jetson` |
| PyTorch | 2.9.1，CUDA 可用，`sm_87` |
| Transformers | 4.57.6 |
| Accelerate / Pillow | 1.12.0 / 12.1.0 |
| 空闲 RAM | 停止模型服务后 available 约 5.9–6.8 GiB |
| Swap | 3.7 GiB zram + 8 GiB 临时文件 swap |

### 5.2 `free`、cache 和 swap 应该如何理解

Linux 会把暂时不用的 RAM 用作 page cache，所以：

- `free` 很小不等于泄漏；
- 优先观察 `available`；
- swap 设备 enabled 不等于已经大量换页；
- 还要观察 `swapon --show` 的 `USED`；
- Jetson 是统一内存架构，CPU/GPU/TensorRT 会竞争同一物理内存，连续大块分配失败不能
  只看 `free` 一项。

早期空闲检查大约是：

```text
Mem total 7.4 GiB, available 5.7 GiB
Swap total 3.7 GiB, used approximately 0
```

Edge-LLM 20 样本运行时则是真实高压状态：RAM 峰值 `7414 MB`，swap 峰值
`2579 MB`。

### 5.3 JetPack / L4T 升级

升级前先备份 NVIDIA apt source，然后把源从 r36.4 切换到 r36.5：

```bash
sudo cp -a /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
  /etc/apt/sources.list.d/nvidia-l4t-apt-source.list.pre-r36.5

sudo sed -i 's/r36\.4/r36.5/g' \
  /etc/apt/sources.list.d/nvidia-l4t-apt-source.list

cat /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
sudo apt update
sudo apt dist-upgrade
```

升级过程中 `nvidia-l4t-bootloader` 曾导致 `dpkg returned an error code (1)`。处理后
相关 NVIDIA 包状态恢复为 `ii`，再执行重启。完整修复终端输出当时没有保存，因此这里
不补写未经证据确认的命令。重启后通过下列命令验收，而不是仅看 apt 返回：

```bash
cat /etc/nv_tegra_release
dpkg -l | grep -E 'nvidia-jetpack|nvidia-l4t-core|nvidia-l4t-bootloader'
/usr/local/cuda/bin/nvcc --version
python3 -c 'import tensorrt as trt; print(trt.__version__)'
```

最终确认 R36.5.0 / JetPack 6.2.2 / CUDA 12.6 / TensorRT 10.3 可用。

## 6. Jetson 项目 Python 环境

### 6.1 为什么使用 `.venv-jetson`

板端原来已有：

```text
/home/ubuntu/project/llm-on-device/.venv
Python 3.12.12
torch 2.9.1+cu126
```

它是另一个项目的环境，而且 torch 是通用 aarch64 CUDA wheel，只包含
`sm_80`、`sm_90`，不包含 Orin Nano 的 `sm_87`。因此没有覆盖原 venv，而是在本仓库
创建：

```text
/home/ubuntu/JetsonVLM/.venv-jetson
```

`.venv-jetson` 是设备相关环境；`pyproject.toml` 和 `uv.lock` 不能可靠表达 JetPack
专用、按 URL 和 SHA 固定的 aarch64 PyTorch wheel，所以板端安装过程单独记录。

### 6.2 创建 venv

系统 `python3.10 -m venv` 因缺少 ensurepip/python3.10-venv 失败，随后使用已有 uv：

```bash
/home/ubuntu/.local/bin/uv venv \
  --clear \
  --seed \
  --no-project \
  --python /usr/bin/python3.10 \
  /home/ubuntu/JetsonVLM/.venv-jetson
```

结果：Python 3.10.12，并预置 pip、setuptools、wheel、packaging。

### 6.3 安装 Jetson PyTorch

固定 wheel：

```text
torch-2.9.1-cp310-cp310-linux_aarch64.whl
bytes  = 228271497
sha256 = 02fde421eabbf62633092de30405ea4d917323c55bea22bfd10dfeb1f1023506
```

原始 URL：

```text
https://pypi.jetson-ai-lab.io/jp6/cu126/+f/02f/de421eabbf626/torch-2.9.1-cp310-cp310-linux_aarch64.whl
```

板端直连过慢，因此 PC 通过已配置代理下载、校验，再用 `scp` 传输。板端安装：

```bash
/home/ubuntu/.local/bin/uv pip install \
  --python /home/ubuntu/JetsonVLM/.venv-jetson/bin/python \
  /home/ubuntu/JetsonVLM/.venv-jetson/torch-2.9.1-cp310-cp310-linux_aarch64.whl
```

### 6.4 Transformers、cuDSS 和 torchvision

```bash
export HTTP_PROXY=http://192.168.137.1:7897
export HTTPS_PROXY=http://192.168.137.1:7897

/home/ubuntu/.local/bin/uv pip install \
  --python /home/ubuntu/JetsonVLM/.venv-jetson/bin/python \
  'transformers==4.57.6' \
  'accelerate==1.12.0' \
  'Pillow==12.1.0' \
  'huggingface-hub==0.36.0'
```

第一次 `import torch` 缺少 `libcudss.so.0`。只补 cuDSS，不安装另一套通用 CUDA：

```bash
/home/ubuntu/.local/bin/uv pip install \
  --no-deps \
  --python /home/ubuntu/JetsonVLM/.venv-jetson/bin/python \
  'nvidia-cudss-cu12==0.4.0.2.post1'
```

运行时动态库路径：

```bash
export JETSON_PY_CUDA_LIB=/home/ubuntu/JetsonVLM/.venv-jetson/lib/python3.10/site-packages/nvidia/cu12/lib
export LD_LIBRARY_PATH=$JETSON_PY_CUDA_LIB:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

run3 又发现 Qwen3-VL 的 `AutoVideoProcessor` 依赖 torchvision，于是安装与 torch 2.9.1
配套的 Jetson wheel：

```bash
/home/ubuntu/.local/bin/uv pip install \
  --python /home/ubuntu/JetsonVLM/.venv-jetson/bin/python \
  'https://pypi.jetson-ai-lab.io/jp6/cu126/+f/d5b/caaf709f11750/torchvision-0.24.1-cp310-cp310-linux_aarch64.whl'
```

### 6.5 检查包版本

```bash
cd /home/ubuntu/JetsonVLM
.venv-jetson/bin/python -m pip list
.venv-jetson/bin/python -m pip show torch torchvision transformers accelerate Pillow
```

最终关键版本：

```text
Python       3.10.12
torch        2.9.1
torchvision  0.24.1
transformers 4.57.6
accelerate   1.12.0
Pillow       12.1.0
tensorrt     10.3.0
nvidia-cudss-cu12 0.4.0.2.post1
```

### 6.6 `sm_87` 真正执行验证

```bash
.venv-jetson/bin/python - <<'PY'
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_arch_list())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
x = torch.ones(1024 * 1024, device="cuda", dtype=torch.float16)
y = (x * 3).sum()
torch.cuda.synchronize()
print(y.item())
PY
```

实测：

```text
torch=2.9.1
cuda_available=True
cuda_runtime=12.6
arch_list=['sm_87']
device=Orin
capability=(8, 7)
kernel_result=inf
```

`inf` 是 FP16 累加溢出，不是 kernel 失败；创建、计算和同步均已实际完成。

## 7. 模型 revision、下载与缓存

### 7.1 为什么固定不可变 commit

如果使用 Hugging Face `main`，上游可能更新 config、tokenizer 或权重，使不同阶段在不知
情的情况下使用不同模型。项目因此固定：

```text
model_id = Qwen/Qwen3-VL-2B-Instruct
revision = 89644892e4d85e24eaac8bacfd4f463576704203
```

该 revision 同时写入 Transformers、Edge-LLM、导出和 LoRA 示例配置。

### 7.2 板端下载命令与位置

```bash
/home/ubuntu/project/llm-on-device/.venv/bin/hf download \
  Qwen/Qwen3-VL-2B-Instruct \
  --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --cache-dir /home/ubuntu/.cache/huggingface/hub
```

snapshot：

```text
/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
```

校验结果：

```text
files                         12/12
bytes                         4266648961/4266648961
.incomplete files             0
model.safetensors SHA-256     与 Hugging Face LFS blob hash 一致
cache size                    约 4.0 GiB
```

板端旧缓存仍保留：

```text
/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B
```

约 1.5 GiB，没有因本项目下载而删除。

### 7.3 为什么 Qwen3-VL-2B 的 ONNX 约 4.6 GiB

2B 参数若绝大部分以 FP16 保存，仅权重理论量级就是：

```text
2 × 10^9 parameters × 2 bytes ≈ 4 GB
```

再加上视觉编码器、图结构、常量和 ONNX external data/sidecar，归档约 4.6 GiB 是正常
数量级。它不是“输入数据量”，也不是 engine 运行时实际只需 4.6 GiB RAM 的保证。

## 8. Transformers FP16：从失败到板端基线

### 8.1 通用单图命令

```bash
cd /home/ubuntu/JetsonVLM

LD_LIBRARY_PATH=/home/ubuntu/JetsonVLM/.venv-jetson/lib/python3.10/site-packages/nvidia/cu12/lib:/usr/local/cuda/lib64 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src \
timeout 900s \
.venv-jetson/bin/python -m parksight_vlm.app.analyze_image \
  --image data/raw/v2-5b48cc8372861680ce69be0d30d05319_1440w.jpg \
  --runtime transformers \
  --backend-revision 'transformers==4.57.6+torch==2.9.1-jetson' \
  --model-revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --precision fp16 \
  --device-map auto \
  --dtype float16 \
  --attn-implementation sdpa
```

若 stdout 重定向到文件，终端没有输出是正常现象；必须同时查看退出码、结果 JSON 和
stderr，不能仅凭屏幕为空判断程序卡住。

### 8.2 run1–run5

| run | 环境 / 修改 | 结果 | 得到的结论 |
| --- | --- | --- | --- |
| 入口初测 | 原 CLI | `--dtype` 等参数无法识别 | 增加参数并传入 runtime options |
| run1 | 原 Python 3.12 通用 torch | NvMap error 12，约 15.25 s | 此结果混有错误 torch wheel，不能判断模型容量 |
| run2 | 增加 CUDA arch 预检 | `dependency_unavailable`，约 7.09 s | 明确发现 torch 不含 `sm_87` |
| run3 | 新 `.venv-jetson` | 缺少 torchvision，12.26 s | 安装 Jetson torchvision 0.24.1 |
| run4 | 依赖补齐 | NvMap/NVML allocator error，12.15 s | 权重加载阶段连续大块分配失败 |
| run5 | `PYTORCH_NO_CUDA_MEMORY_CACHING=1` | `out_of_memory`，10.53 s | 不是单纯 caching allocator 预留导致 |

run5 遥测：RAM 峰值约 `4005 MiB`，swap 为 0，最小 lfb 为 `2 × 4 MiB`，GPU 利用率
仍为 0%。退出后 available 约 5.9 GiB。结合当时 R36.4.7 的 NvMap 问题，不能得出
“2B FP16 必然无法放入 8GB”的结论。

### 8.3 run6–run8：业务链路逐步修复

升级到 R36.5.0 后：

| run | 结果 | 原因 / 修复 |
| --- | --- | --- |
| run6 | `runtime_error: string indices must be integers` | Qwen3-VL 多模态 system message 结构不符合 processor 期望 |
| run7 | 模型已生成 JSON，但严格解析失败 | `events` 使用中文自由文本，`evidence`/`driver_advice` 还是字符串 |
| run8 | `failure=null`，严格 JSON 成功 | 修复消息格式并加强 prompt 的英文 snake_case 枚举约束 |

run7 的错误已经不是 GPU、torch 或模型加载问题，而是输出契约问题。它说明一条推理链
可以在“模型确实跑了”的情况下仍然被业务层判定失败。

## 9. PS2.0 数据集与 20 样本标注

### 9.1 为什么选 AVM 鸟瞰图

调研过 WoodScape、FPD、AVM-SLAM、Tongji PS2.0、nuScenes 等数据源。为了尽快打通
完整推理与评测链路，第一轮选择 Tongji PS2.0：

- 输入已经是四鱼眼拼接的 `600 × 600` AVM 鸟瞰图；
- 包含室内、白天、雨天、阴影、路灯、斜列车位；
- 与低速停车场景接近；
- 但只提供停车位几何标注，没有本项目所需风险、证据和驾驶建议标注。

数据规模与许可边界见 [`surround-view-datasets.md`](surround-view-datasets.md)。原始数据
的下载终端命令没有完整保存，本文不伪造；原始图片及 `.mat` 只作为本地研究输入，
正式再分发前需重新核对许可证。

### 9.2 pilot 组成

从官方 testing 分类目录分层选择 20 张：

| 场景 | 数量 |
| --- | ---: |
| indoor parking lot | 3 |
| outdoor normal daylight | 4 |
| outdoor rainy | 3 |
| outdoor shadow | 4 |
| outdoor slanted | 3 |
| outdoor street light | 3 |
| 合计 | 20 |

对应文件：

```text
data/manifests/ps20_pilot_v1.jsonl
data/annotations/ps20_pilot_v1.jsonl
data/raw/ps2.0/pilot/
configs/studies/jetson_transformers_fp16_ps20_pilot.json
configs/studies/jetson_edgellm_fp16_ps20_pilot.json
```

标注是根据图片可见内容人工生成的项目 JSONL，不是 PS2.0 原始标签的简单转换。由于
图片没有车辆挡位、规划轨迹和目标车位，标注采用保守边界：不凭空推断运动意图；没有
轨迹时不轻易标 `near_maneuver_path`；没有目标车位时不标
`parking_space_conflict`。

该 pilot 只有单人第一轮目视标注，目前全部属于冻结 `test`，不能直接拿去训练。

## 10. Jetson Transformers FP16 20 样本基线

### 10.1 执行命令

```bash
cd /home/ubuntu/JetsonVLM

LD_LIBRARY_PATH=/home/ubuntu/JetsonVLM/.venv-jetson/lib/python3.10/site-packages/nvidia/cu12/lib:/usr/local/cuda/lib64 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src \
timeout 1800s \
.venv-jetson/bin/python -m parksight_vlm.app.run_study \
  --config configs/studies/jetson_transformers_fp16_ps20_pilot.json
```

同时以 1 秒间隔运行 `tegrastats`。报告环境快照确认：

```text
L4T R36.5.0
Python 3.10.12
torch 2.9.1
transformers 4.57.6
Pillow 12.1.0
power_mode=15W_MODE_0
```

### 10.2 质量结果

| 指标 | 结果 |
| --- | ---: |
| 样本数 / 成功数 | 20 / 20 |
| 严格 JSON 有效率 | 1.000 |
| 风险等级准确率 | 0.350 |
| 事件 micro precision | 0.280 |
| 事件 micro recall | 0.4375 |
| 事件 micro F1 | 0.3415 |
| 不安全建议率 | 0.000 |
| 运行时失败 | 0 |

主要误差：20 张全部预测为 `low`；18 张预测 `narrow_passage`，7 张额外预测
`vehicle_near_maneuver_path`，没有预测 `visibility_occlusion`。这表明 JSON 契约已经
成立，但业务判断质量较低。

### 10.3 性能与资源

| 指标 | 结果 |
| --- | ---: |
| 冷启动端到端 | 31,081.31 ms |
| 预处理 p50 / p90 / p99 | 27.50 / 31.30 / 86.93 ms |
| 模型生成 p50 / p90 / p99 | 9,354.70 / 14,086.91 / 14,972.91 ms |
| 端到端 p50 / p90 / p99 | 9,384.95 / 14,117.36 / 27,939.48 ms |
| 进程峰值内存 | 4,176.72 MB |
| 系统 RAM 峰值 | 6,398 MB |
| swap 峰值 | 501 MB |
| GPU 利用率峰值 | 99% |
| GPU 温度峰值 | 61.781 °C |
| `VDD_IN` 瞬时峰值 | 12.655 W |
| `VDD_IN` 区间平均 | 10.115 W |

原始证据：

```text
reports/jetson_transformers_fp16_ps20_pilot.json
reports/jetson_transformers_fp16_ps20_pilot_20260802_tegrastats.log
reports/jetson_transformers_fp16_ps20_pilot_20260802_stdout.log
reports/jetson_transformers_fp16_ps20_pilot_20260802_stderr.log
```

## 11. GPU 服务器导出 TensorRT Edge-LLM ONNX

### 11.1 固定环境

| 项目 | 实测值 |
| --- | --- |
| 系统 | Ubuntu 22.04.5 LTS，x86-64 |
| GPU | NVIDIA GeForce RTX 4090 D，24 GiB |
| 驱动 / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| 导出 Python | 3.12.3，独立 `.venv-export` |
| PyTorch | `2.12.0+cu126` |
| TensorRT Edge-LLM | v0.9.1，commit `7f061f21f0a581ba234a1e233c9315b89d8e47d6` |
| Transformers / ONNX / ONNX Script | 5.9.0 / 1.19.0 / 0.7.0 |

服务器没有系统 `nvcc`，但 PyTorch CUDA FP16 matrix smoke test 通过；Python ONNX 导出
本身不要求在本次服务器上编译 Edge-LLM C++ runtime。

### 11.2 固定源码和模型

```bash
cd /root/autodl-tmp
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git checkout --detach 7f061f21f0a581ba234a1e233c9315b89d8e47d6
git submodule update --init --recursive
git rev-parse HEAD
```

模型权重：

```text
model.safetensors bytes  = 4255140312
model.safetensors sha256 = 7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0
```

该 revision 是单个 `model.safetensors`，没有
`model.safetensors.index.json`。导出 flow 的 required inputs 因此修正为真实文件。

### 11.3 dry-run 与正式导出

```bash
export PYTHONPATH=/root/autodl-tmp/JetsonVLM/src
export PATH=/root/autodl-tmp/TensorRT-Edge-LLM/.venv-export/bin:$PATH

python scripts/export_model.py \
  --config configs/flows/export_qwen3_vl_2b_fp16.json

python scripts/export_model.py \
  --config configs/flows/export_qwen3_vl_2b_fp16.json \
  --execute
```

结果：

```text
dry-run ready=true
missing_inputs=[]
preexisting_outputs=[]
正式执行 exit_code=0
flow status=succeeded
开始 2026-08-04T15:12:59.286487+00:00
结束 2026-08-04T15:15:14.059600+00:00
```

输出包含 LLM 和视觉编码器的 ONNX、external data、tokenizer/chat template、视觉
preprocessor/config 等共 11 个文件。

归档：

```text
artifacts/transfers/qwen3_vl_2b_fp16_export_89644892.tar
bytes  = 4890972160
sha256 = 7e00cd92099ff9f35ed600e68682630b70c87cc58e1311874a15002d63fc4e45
```

服务器、本机和 Jetson 的归档哈希一致，内部 11 个文件逐项 SHA-256 均为 `OK`。

## 12. Jetson 编译 TensorRT Edge-LLM

### 12.1 固定上游身份

```text
TensorRT Edge-LLM v0.9.1
commit 7f061f21f0a581ba234a1e233c9315b89d8e47d6
```

板端源码位置：

```text
/home/ubuntu/TensorRT-Edge-LLM
```

离线补齐 NVTX、googletest、nlohmann/json 子模块，使用 `.venv-jetson` 中的 CMake
3.31.6 和 pybind11 2.13.6 编译。编译成功标准：

```text
build/libNvInfer_edgellm_plugin.so
build/examples/llm/llm_build
build/examples/llm/llm_inference
build/examples/multimodal/visual_build
build/pybind/_edgellm_runtime.cpython-310-aarch64-linux-gnu.so
```

`ldd` 没有缺失依赖，以下 Python 导入通过：

```text
_edgellm_runtime.LLMRuntime
experimental.server.engine.LLM
```

### 12.2 为什么需要项目补丁

上游 commit 保持不变，JetPack 6.2.2 / TensorRT 10.3 兼容改动以补丁形式保存在：

| 补丁 | 作用 |
| --- | --- |
| `0001-tensorrt-10.3-stream-reader-compat.patch` | TensorRT 10.3 stream reader 接口兼容 |
| `0002-jetpack-6-cutedsl-compat.patch` | JetPack 6 的 CuteDSL 兼容 |
| `0003-tensorrt-10.3-disable-fp4-plugin-formats.patch` | 禁用 TensorRT 10.3 不支持的 FP4 plugin format |
| `0004-jetpack-6-propagate-cutedsl-shim.patch` | 将 CuteDSL shim 传入相关目标 |
| `0005-log-fmha-cubin-metadata.patch` | 输出 FMHA CUBIN 元数据便于定位 |
| `0006-skip-rejected-unused-fmha-cubins.patch` | 仅跳过驱动明确拒绝且当前模型不用的 CUBIN |
| `0007-configurable-builder-memory.patch` | 允许限制 builder workspace / 搜索内存 |
| `0008-configurable-weight-streaming.patch` | 构建阶段启用可配置 weight streaming |
| `0009-configurable-runtime-weight-streaming-budget.patch` | 创建 context 前设置运行时权重流式预算 |

### 12.3 FMHA CUBIN 问题

LLM ONNX 的第一个 `AttentionPlugin` 曾因 `CUDA_ERROR_INVALID_IMAGE` 创建失败。逐条验证
11 个 SM87 FMHA CUBIN 后：

- 9 个能够加载；
- 只有两个 `head_size=256 + custom_mask` 资产被驱动拒绝；
- Qwen3-VL-2B 实际 `head_size=128`。

因此补丁只在 CUDA 驱动明确返回 `INVALID_IMAGE` 时跳过未使用资产。随后 32 层
`AttentionPlugin` 均成功创建，ONNX 解析和图优化继续完成。

## 13. Jetson 构建 FP16 engine

### 13.1 为什么拆成 visual 和 LLM 两个 flow

最初组合构建时，LLM 阶段 OOM 会覆盖视觉阶段的成功/失败边界。构建入口因此支持：

```text
--component llm
--component visual
--component both
```

并提供独立配置：

```text
configs/flows/build_qwen3_vl_2b_fp16_visual_engine.json
configs/flows/build_qwen3_vl_2b_fp16_llm_engine.json
```

### 13.2 visual engine

```bash
cd /home/ubuntu/JetsonVLM
PYTHONPATH=src .venv-jetson/bin/python scripts/build_engine.py \
  --config configs/flows/build_qwen3_vl_2b_fp16_visual_engine.json

PYTHONPATH=src .venv-jetson/bin/python scripts/build_engine.py \
  --config configs/flows/build_qwen3_vl_2b_fp16_visual_engine.json \
  --execute
```

结果：

```text
status        = succeeded
build time    = 33.991 s
TRT GPU peak  = 789 MiB
TRT CPU peak  = 4417 MiB
visual.engine bytes  = 824000540
visual.engine sha256 = 3c6b4cce682e021b09c066d0e325335e31ef9edbf613c754be586035c26f5c2f
```

### 13.3 LLM 构建 OOM 与临时 swap

六次失败 record/log 依次排查了：

1. plugin 路径；
2. FMHA CUBIN；
3. builder 优化等级；
4. workspace 限制；
5. headless/分阶段构建；
6. weight streaming 构建期内存。

内核日志证明 `llm_build` 在 8 GiB 统一内存和约 3.7 GiB zram 用尽后被 OOM killer
终止；另一次 TensorRT 明确记录额外 `3441150208` 字节分配失败。

最终构建参数：

```text
maxBatchSize       = 1
maxInputLen        = 1024
maxKVCacheCapacity = 2048
workspace limit    = 1024 MiB
optimization level = 0
weight streaming   = enabled
```

创建 8 GiB 临时文件 swap 后启用：

```bash
sudo swapon --priority 1 /home/ubuntu/parksight-build.swap
swapon --show
free -h
```

如果再次执行出现：

```text
insecure file owner 1000, 0 (root) suggested
swapon failed: Device or resource busy
```

含义是：

- owner 警告建议文件归 root；
- `Device or resource busy` 表示该 swap 已经 active，重复启用失败；
- 应通过 `/proc/swaps` 或 `swapon --show` 判断真实状态；
- 该 swap 没有写入 `/etc/fstab`，重启后不会自动启用。

正式 LLM 构建：

```bash
cd /home/ubuntu/JetsonVLM
PYTHONPATH=src .venv-jetson/bin/python scripts/build_engine.py \
  --config configs/flows/build_qwen3_vl_2b_fp16_llm_engine.json \
  --execute
```

结果：

```text
flow_id       = build_qwen3_vl_2b_fp16_llm_7f061f21
status        = succeeded
started_at    = 2026-08-10T00:36:48.769548+00:00
finished_at   = 2026-08-10T00:40:37.732676+00:00
llm.engine bytes  = 3453798316
llm.engine sha256 = cbdf0300bf406dfbbcd06d47435c699c26403139d6bdd06b473ba00576583013
```

## 14. Edge-LLM 双 engine 服务与运行时 OOM 修复

### 14.1 第一次服务启动失败

默认相对 plugin 路径落到 ParkSight 工作目录，导致 `AttentionPlugin` 未注册。修复方式
是设置绝对路径：

```bash
export EDGELLM_PLUGIN_PATH=/home/ubuntu/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
```

### 14.2 第二次服务启动失败

engine 可以反序列化，但 TensorRT 默认关闭 weight streaming，尝试让
`3441150208` 字节 streamable weights 一次性驻留 GPU，随后 OOM。

`0009` 补丁在创建 `IExecutionContext` 前读取
`EDGELLM_WEIGHT_STREAMING_BUDGET_BYTES`，并调用：

```cpp
ICudaEngine::setWeightStreamingBudgetV2(...)
```

只增量编译 Python binding/runtime：

```bash
cd /home/ubuntu/TensorRT-Edge-LLM
/home/ubuntu/JetsonVLM/.venv-jetson/bin/cmake --build build \
  --target _edgellm_runtime --parallel 1
```

### 14.3 成功启动命令

```bash
export PYTHONPATH=/home/ubuntu/TensorRT-Edge-LLM:/home/ubuntu/JetsonVLM/src
export BUILD_DIR=/home/ubuntu/TensorRT-Edge-LLM/build
export EDGELLM_PLUGIN_PATH=/home/ubuntu/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
export EDGELLM_WEIGHT_STREAMING_BUDGET_BYTES=0
export LD_LIBRARY_PATH=/home/ubuntu/JetsonVLM/.venv-jetson/lib/python3.10/site-packages/nvidia/cu12/lib:/home/ubuntu/TensorRT-Edge-LLM/build:/usr/local/cuda/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu

cd /home/ubuntu/JetsonVLM
.venv-jetson/bin/python scripts/serve_edgellm.py \
  --engine-root artifacts/engines/qwen3_vl_2b_fp16 \
  --weight-streaming-budget-bytes 0 \
  --host 127.0.0.1 \
  --port 8000
```

成功日志：

```text
requested=0 actual=0 streamable=3441150208 scratch=1244660224 bytes
Base EngineExecutor successfully loaded
Vision runner successfully initialized
Setup shared execution context memory: 1356027904 bytes
Successfully captured decoding CUDA graphs
Uvicorn running on http://127.0.0.1:8000
```

健康检查：

```bash
curl --fail http://127.0.0.1:8000/health
```

返回 HTTP 200、`status=healthy`。

## 15. Edge-LLM 单图和 20 样本验收

### 15.1 单图命令

```bash
cd /home/ubuntu/JetsonVLM
PYTHONPATH=src .venv-jetson/bin/python -m parksight_vlm.app.analyze_image \
  --image data/raw/ps2.0/pilot/indoor/001.jpg \
  --workload configs/workloads/parking_risk_v1.json \
  --runtime tensorrt_edge_llm_http \
  --backend-revision 7f061f21f0a581ba234a1e233c9315b89d8e47d6 \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --model-revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --adapter-revision edge-http-v2 \
  --precision fp16 \
  --edge-url http://127.0.0.1:8000
```

实测 HTTP 200、57 token、端到端 `38943.90 ms`。模型生成了 JSON，但把
`events`、`evidence`、`driver_advice` 输出成字符串，严格解析结果为：

```text
category       = json_parse_error
message        = events must be an array
exception_type = AssessmentValidationError
```

这证明图像输入、视觉 engine、LLM engine、生成和 HTTP Adapter 都执行了；但业务输出
不合格。

### 15.2 20 样本命令

```bash
cd /home/ubuntu/JetsonVLM
tegrastats --interval 1000 \
  > reports/runtime/jetson_edgellm_fp16_ps20_pilot.tegrastats.log &

PYTHONPATH=src .venv-jetson/bin/python -m parksight_vlm.app.run_study \
  --config configs/studies/jetson_edgellm_fp16_ps20_pilot.json
```

结果：

| 指标 | 结果 |
| --- | ---: |
| 样本数 | 20 |
| 后端完成 | 20/20 |
| 严格 schema 有效 | 0/20 |
| 失败汇总 | `json_parse_error: 20` |
| 端到端 min / mean / max | 38.827 / 44.000 / 56.410 s |
| 端到端 p50 / p90 / p99 | 41.758 / 50.612 / 55.419 s |
| 输出 token 总数 / 均值 | 1299 / 64.95 |
| 聚合端到端输出速率 | 1.476 token/s |

874 条 `tegrastats`：

| 资源指标 | 结果 |
| --- | ---: |
| RAM mean / peak | 7409.29 / 7414 MB |
| swap mean / peak | 2561.64 / 2579 MB |
| GPU 利用率 mean / max | 98.79% / 99% |
| GPU 温度 mean / max | 60.67 / 62.50 °C |
| `VDD_IN` mean / max | 10.11 / 10.76 W |

派生摘要：

```bash
PYTHONPATH=src python scripts/summarize_jetson_study.py \
  --study-report reports/jetson_edgellm_fp16_ps20_pilot.json \
  --tegrastats reports/runtime/jetson_edgellm_fp16_ps20_pilot.tegrastats.log \
  --output reports/jetson_edgellm_fp16_ps20_pilot.runtime-summary.json
```

这个脚本特意把“后端完成”和“严格 schema 成功”分开，避免把 HTTP 200 误写成业务
正确。

## 16. Transformers 与 Edge-LLM 的同机比较

两份报告共同使用：

```text
model revision    89644892e4d85e24eaac8bacfd4f463576704203
workload identity parking_risk_v1@sha256:8350ace4574f8aa154319f7136ef831003d4dcc074ef20b74c1b419d69a2a493
dataset           ps20_pilot_v1 / test / 20 samples
power mode        15W_MODE_0
Jetson L4T        R36.5.0
precision         fp16
```

| 指标 | Transformers FP16 | Edge-LLM FP16 |
| --- | ---: | ---: |
| 后端完成 | 20/20 | 20/20 |
| 严格 JSON 有效 | 20/20 | 0/20 |
| E2E p50 | 9.385 s | 41.758 s |
| E2E p90 | 14.117 s | 50.612 s |
| E2E p99 | 27.939 s | 55.419 s |
| RAM 峰值 | 6398 MB | 7414 MB |
| swap 峰值 | 501 MB | 2579 MB |
| 平均输入功耗 | 10.115 W | 10.110 W |

按当前数据计算，Edge-LLM 的 p50/p90/p99 分别约为 Transformers 的
`4.45× / 3.59× / 1.98×`，即当前实现没有加速，反而更慢。主要背景是 LLM engine
必须使用 `weight-streaming-budget=0` 才能在 8 GB 设备上加载，权重流式访问和 1.24 GB
scratch 带来明显代价。

质量结果也不能直接比较为“模型变差”：两边模型权重相同，但 Edge-LLM 的 prompt/chat
template、请求格式、解码行为或输出约束仍可能存在适配差异。下一轮应先定位 20/20
数组字段变字符串的接口问题，再讨论模型质量和加速。

## 17. 证据文件索引

### 17.1 Transformers

```text
reports/smoke/jetson_transformers_fp16_20260728_run3.json
reports/smoke/jetson_transformers_fp16_20260728_run4.json
reports/smoke/jetson_transformers_fp16_20260728_run5.json
reports/smoke/jetson_transformers_fp16_20260802_run6.json
reports/smoke/jetson_transformers_fp16_20260802_run7.json
reports/smoke/jetson_transformers_fp16_20260802_run8.json
reports/jetson_transformers_fp16_ps20_pilot.json
reports/jetson_transformers_fp16_ps20_pilot_20260802_tegrastats.log
```

### 17.2 ONNX 与 engine

```text
reports/flows/export_qwen3_vl_2b_fp16.json
reports/flows/export_qwen3_vl_2b_fp16.log
reports/jetson-flows-20260810/flows/
reports/jetson-runtime-20260810/build_qwen3_vl_2b_fp16_llm_engine.json
reports/jetson-runtime-20260810/build_qwen3_vl_2b_fp16_llm_engine.log
```

### 17.3 Edge-LLM runtime

```text
reports/jetson-runtime-20260810/edgellm_fp16_server_run3.log
reports/jetson-runtime-20260810/jetson_edgellm_fp16_ps20-indoor-001.json
reports/jetson-runtime-20260810/jetson_edgellm_fp16_ps20_pilot.json
reports/jetson-runtime-20260810/jetson_edgellm_fp16_ps20_pilot.tegrastats.log
reports/jetson-runtime-20260810/jetson_edgellm_fp16_ps20_pilot.runtime-summary.json
```

这些大型/运行时证据大多被 `.gitignore` 忽略；Git 中保存代码、配置、补丁和文档，原始
运行证据保存在本机。

## 18. 当前测试与验证状态

当前代码修改后的标准检查：

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests
python -m compileall -q src scripts
git diff --check
```

最近一次完整结果：

```text
Ran 38 tests
OK
compileall: passed
git diff --check: passed
```

这些是无硬件验证，不能替代 Jetson engine 构建和真实推理；本项目同时保存了第 10、
13、14、15 节对应的板端实测证据。

## 19. 尚未完成的工作

### 19.1 Edge-LLM 输出契约修复

最高优先级。需要逐层比较 Transformers 与 Edge-LLM 的：

1. system/user message 构造；
2. chat template 渲染文本；
3. 图片 token 和视觉 embedding 位置；
4. sampling/generation 参数；
5. tokenizer 解码与 stop 条件；
6. 是否支持 grammar/constrained decoding；
7. 原始输出为何把数组序列化为字符串。

成功标准不是只返回 HTTP 200，而是相同 20 样本的严格 JSON 有效率恢复到可比较水平。

### 19.2 LoRA

当前只有 20 个冻结 `test` 样本，没有独立 `train`/`validation`。示例配置仍引用不存在
的：

```text
data/manifests/parking_risk_v1.jsonl
data/annotations/parking_risk_v1.jsonl
```

训练 flow command 仍是：

```text
replace-with-reviewed-training-command
```

因此目前没有执行 LoRA、没有 adapter、没有合并模型，也没有训练后复测。下一步必须从
独立来源组建立训练/验证数据，不能泄漏冻结 test 集。

### 19.3 INT4

当前没有：

- 审核后的 INT4/AWQ 方法；
- 校准数据；
- INT4 构建配置；
- INT4 engine；
- 板端质量/性能报告。

FP16 已证明优化动机：3.44 GB streamable weights、1.24 GB scratch、RAM 峰值 7.4 GB、
swap 峰值 2.58 GB、p50 41.76 s。但这些不能被写成 INT4 已完成。

### 19.4 服务器正确性参考

服务器 ONNX 导出已经完成，但相同 `ps20_pilot_v1` 的服务器 Transformers StudyReport
还没有形成。它用于正确性和误差分析，不用于计算 Jetson 加速比。

## 20. 建议的下一步顺序

1. 修复 Edge-LLM prompt/chat template 或响应适配，使严格 JSON 先通过；
2. 在 20 样本上重跑 Edge-LLM FP16，得到可比较质量报告；
3. 记录不同 weight-streaming budget 的可加载上限与延迟，确认当前慢的直接来源；
4. 从 PS2.0 或更合适的原始环视数据建立独立 train/validation，并二次复核标注；
5. 在 4090 D 上执行 LoRA，合并后先做服务器正确性评测；
6. 将合并模型重新导出，在 Jetson 复测；
7. 选定 Edge-LLM 支持的 INT4/AWQ 路径和校准集；
8. 构建 INT4 engine，与 Transformers FP16、Edge-LLM FP16 做同机对照；
9. 最终报告同时给出质量、p50/p90/p99、tokens/s、RAM、swap、功耗、温度和失败样例。

## 21. 本阶段形成的核心认识

- 项目不是简单调用一次 `model.generate()`，而是把输入、输出契约、模型身份、Runtime、
  失败记录和评测绑定成可复现链路。
- Hugging Face commit、Edge-LLM commit、workload SHA 和数据集 revision 都属于实验身份。
- ONNX 是模型交换表示，TensorRT engine 才是针对目标设备构建的执行产物。
- x86 GPU 服务器导出的 ONNX 可以传到 Jetson；4090 上构建的 engine 不能直接代替
  Jetson engine。
- HTTP 200 只证明后端完成，严格 JSON 校验通过才证明业务输出成功。
- TensorRT 不天然等于更快。内存不足导致 0-byte weight streaming budget 时，实际性能
  可能显著低于 Transformers。
- `free`、swap enabled、GPU 利用率和 OOM 必须结合统一内存、连续块和内核日志解释。
- 失败也是实验结果，但必须保存失败类别、命令、环境和原始日志，不能只写“跑不起来”。

## 22. 2026-08-11：Prompt 契约诊断与第一处对齐修改

### 22.1 为什么先做 Prompt 契约诊断

2026-08-10 的两份同机报告已经证明：

```text
Transformers FP16  20/20 严格 JSON 有效
Edge-LLM FP16      20/20 HTTP 后端完成，但 0/20 严格 JSON 有效
```

Edge-LLM 的典型原始输出把 `events`、`evidence` 和 `driver_advice` 输出成普通字符串，
而不是 workload 明确要求的 JSON 数组。在这个问题解决前直接进行 LoRA 或 INT4，会把
消息格式、chat template、模型能力和量化误差混为一谈。因此当前优先级是证明两个
Runtime 最终使用的消息和 Prompt 是否一致。

### 22.2 发现的第一处差异

修改前，两条 Runtime 的 system message 结构不同：

```python
# Transformers
{
    "role": "system",
    "content": [
        {"type": "text", "text": workload.system_prompt}
    ],
}

# Edge-LLM HTTP
{
    "role": "system",
    "content": workload.system_prompt,
}
```

Edge 的 user message 已经使用多模态 content 数组，但 system message 是普通字符串。
该差异不等于已经证明是 0/20 JSON 失败的唯一原因，但它是两条运行路径中可以静态确认
的协议差异，因此先执行最小对齐。

修改后的 Edge system message 与 Transformers 一致，使用：

```python
{
    "role": "system",
    "content": [
        {"type": "text", "text": workload.system_prompt}
    ],
}
```

### 22.3 TDD 过程

本次选择两个公开测试 seam：

1. `EdgeLlmHttpBackend.generate()`：观察真实发送到 HTTP 系统边界的请求 JSON；
2. `scripts/inspect_prompt_contract.py`：观察命令行输出的诊断 JSON。

第一个 red 测试先要求 Edge system content 为文本内容数组，实际失败输出证明旧代码仍
返回字符串。只修改 Edge 请求结构后，该测试变为 green。

第二个 red 测试先导入尚不存在的 `parksight_vlm.inference.prompt_contract`，结果为：

```text
ModuleNotFoundError: No module named 'parksight_vlm.inference.prompt_contract'
```

实现报告构造器后变为 green。第三个 red 测试要求命令行入口，先得到：

```text
ModuleNotFoundError: No module named 'scripts.inspect_prompt_contract'
```

增加 CLI 后，定向测试和全量测试均通过。

### 22.4 新增和修改的文件

| 文件 | 内容 |
| --- | --- |
| `src/parksight_vlm/inference/edge_llm.py` | system message 改为 typed content array；增加 `build_request_payload()`，诊断和真实请求复用同一实现 |
| `src/parksight_vlm/inference/transformers.py` | 将消息构造函数公开为 `build_qwen3_vl_chat_messages()`，诊断复用真实 Transformers 消息结构 |
| `src/parksight_vlm/inference/prompt_contract.py` | 生成 Prompt 契约诊断报告 |
| `scripts/inspect_prompt_contract.py` | Jetson 可直接执行的诊断 CLI |
| `tests/inference/test_prompt_contract.py` | 报告和 CLI 行为测试 |
| `tests/inference/test_runtime.py` | Edge system content 与 Transformers typed message 回归测试 |
| `tests/fixtures/prompt_contract/` | 不依赖真实模型权重的 template/model 测试 fixture |

### 22.5 诊断报告包含什么

报告 schema：

```text
parksight_prompt_contract_report_v1
```

主要字段：

| 字段 | 用途 |
| --- | --- |
| `workload_identity` | 确认 prompt/schema/生成参数没有漂移 |
| `rendered_user_prompt` | 查看完整枚举、数组类型和禁止翻译约束 |
| `transformers_messages` | Transformers 路径实际使用的结构化消息 |
| `transformers_rendered_prompt` | `AutoProcessor.apply_chat_template(tokenize=False)` 渲染出的最终文本 |
| `edge_http_request` | Edge Adapter 真正会发送的完整 HTTP payload |
| `processed_chat_template.path` | Edge engine 使用的已处理模板位置 |
| `processed_chat_template.bytes` | 模板文件大小 |
| `processed_chat_template.sha256` | 模板身份，便于跨设备和重建前后比较 |
| `processed_chat_template.payload` | 模板 JSON 原始内容 |
| `message_contract.messages_equal` | 两条路径的结构化 messages 在诊断表示中是否一致 |
| `message_contract.system_content_type` | system content 是否为 `array` |
| `message_contract.user_content_types` | 预期为 `["image", "text"]` |

该脚本只加载 processor/tokenizer，不加载 Qwen3-VL 完整模型，也不执行 GPU 推理。

### 22.6 本机验证结果

```powershell
$env:PYTHONPATH = 'src'
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests
& '.\.venv\Scripts\python.exe' scripts\inspect_prompt_contract.py --help
```

结果：

```text
Ran 40 tests in 0.121s
OK
inspect_prompt_contract.py --help: exit code 0
```

这只证明请求构造和诊断入口在无硬件环境中成立，不证明 Edge-LLM 板端 JSON 已经修复。

### 22.7 将修改同步到 Jetson

按照当前协作方式，PC 保留 Git 与个人记录，改动文件可以直接 `scp` 到板端。以下命令
在 Windows PowerShell 5.1 中执行：

```powershell
Set-Location -LiteralPath 'C:\Users\22122\Desktop\JetsonVLM'

scp `
  'src\parksight_vlm\inference\edge_llm.py' `
  'src\parksight_vlm\inference\transformers.py' `
  'src\parksight_vlm\inference\prompt_contract.py' `
  ubuntu@192.168.137.187:/home/ubuntu/JetsonVLM/src/parksight_vlm/inference/

scp `
  'scripts\inspect_prompt_contract.py' `
  ubuntu@192.168.137.187:/home/ubuntu/JetsonVLM/scripts/
```

预期：两条 `scp` 命令均返回 0；板端对应四个文件的修改时间更新。

### 22.8 Jetson 生成 Prompt 契约报告

以下命令在 SSH 登录 Jetson 后的 Bash 中执行：

```bash
cd /home/ubuntu/JetsonVLM

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src \
.venv-jetson/bin/python scripts/inspect_prompt_contract.py \
  --workload configs/workloads/parking_risk_v1.json \
  --image data/raw/ps2.0/pilot/indoor/001.jpg \
  --model-source /home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203 \
  --processed-chat-template artifacts/engines/qwen3_vl_2b_fp16/llm/processed_chat_template.json \
  --output reports/prompt-contract/qwen3_vl_2b_fp16_20260811.json
```

目的：在不运行 engine 的情况下，把 Transformers 渲染文本、Edge HTTP payload 和
engine chat template 身份放进同一份 JSON。

第一层成功标准：

```text
命令 exit code = 0
schema_version = parksight_prompt_contract_report_v1
workload_identity = parking_risk_v1@sha256:8350ace4574f8aa154319f7136ef831003d4dcc074ef20b74c1b419d69a2a493
message_contract.messages_equal = true
message_contract.system_content_type = array
message_contract.user_content_types = ["image", "text"]
```

还要人工检查 `transformers_rendered_prompt` 是否包含：

```text
events 必须是 JSON 数组
evidence 必须是 JSON 字符串数组
driver_advice 必须是非空 JSON 数组
不得把数组字段输出为普通字符串
```

如果以上内容缺失，问题仍在 processor/chat template 渲染之前；如果全部存在但 Edge
推理仍输出字符串，则下一层排查重点转到 Edge-LLM C++ server 对
`processed_chat_template.json` 的解释、stop/sampling 参数和 token 解码。

### 22.9 修改后的单图验收

诊断报告通过后，保持 2026-08-10 已验证的 Edge-LLM server 启动方式不变，只重新运行
单图命令：

```bash
cd /home/ubuntu/JetsonVLM

PYTHONPATH=src .venv-jetson/bin/python -m parksight_vlm.app.analyze_image \
  --image data/raw/ps2.0/pilot/indoor/001.jpg \
  --workload configs/workloads/parking_risk_v1.json \
  --runtime tensorrt_edge_llm_http \
  --backend-revision 7f061f21f0a581ba234a1e233c9315b89d8e47d6 \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --model-revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --adapter-revision edge-http-system-content-array-v1 \
  --precision fp16 \
  --edge-url http://127.0.0.1:8000
```

第二层成功标准：

```text
HTTP 200
assessment != null
failure = null
events 是 JSON array
evidence 是 JSON array
driver_advice 是 JSON array
所有枚举值合法
```

先对同一张图片连续运行三次。只有 3/3 严格 JSON 通过，才扩展到 5 张小样本；5/5
通过后再重新运行 20 样本 Study。当前修改尚未形成这部分板端结果，因此不能把它写成
“Edge-LLM JSON 问题已解决”。

## 23. 2026-08-12：Jetson Prompt 契约实测与 Edge 服务启动排障

### 23.1 本次目标和执行边界

本次在 Jetson 上执行以下工作：

1. 将 Prompt 契约相关的四个实现文件从 PC 同步到板端；
2. 在固定 Qwen3-VL revision 和固定 workload 上生成真实 Prompt 契约报告；
3. 启动已有 TensorRT Edge-LLM FP16 engine；
4. 若服务健康则执行同一张图片三次严格 JSON 验收；
5. 若服务无法启动，则保留完整日志并定位加载阶段、内存状态和失败边界。

本次没有执行 Git commit/push，也没有覆盖原有 engine。板端临时同步仍采用 `scp`，PC
仓库继续保存代码与个人执行记录。

### 23.2 文件同步与身份校验

同步文件：

```text
src/parksight_vlm/inference/edge_llm.py
src/parksight_vlm/inference/transformers.py
src/parksight_vlm/inference/prompt_contract.py
scripts/inspect_prompt_contract.py
```

PC 与 Jetson 的 SHA-256 逐文件一致：

```text
c9f9fd6069f8b17ff12c6cb65c9c6b69184f3408fc8c5295d0922fb53339acf2  edge_llm.py
04a6a8911d829296940541154c45be2b73669e832cf21576e0461dbaf7030d92  transformers.py
b3333ccb12e8cc039b1fa8e6ead23018678fd714745518086d5b823568137d5e  prompt_contract.py
984a0f904c2aaa309dc07895c2fb228a1709f801d0478a27a2a48e748e12785a  inspect_prompt_contract.py
```

因此后续板端报告和服务请求确实使用本次修改后的实现，而不是 Jetson 上的旧文件。

### 23.3 Prompt 契约板端报告

实际命令：

```bash
cd /home/ubuntu/JetsonVLM

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=/home/ubuntu/JetsonVLM/src \
LD_LIBRARY_PATH=/home/ubuntu/JetsonVLM/.venv-jetson/lib/python3.10/site-packages/nvidia/cu12/lib:/usr/local/cuda/lib64 \
/home/ubuntu/JetsonVLM/.venv-jetson/bin/python \
  /home/ubuntu/JetsonVLM/scripts/inspect_prompt_contract.py \
  --workload /home/ubuntu/JetsonVLM/configs/workloads/parking_risk_v1.json \
  --image /home/ubuntu/JetsonVLM/data/raw/ps2.0/pilot/indoor/001.jpg \
  --model-source /home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203 \
  --processed-chat-template /home/ubuntu/JetsonVLM/artifacts/engines/qwen3_vl_2b_fp16/llm/processed_chat_template.json \
  --output /home/ubuntu/JetsonVLM/reports/prompt-contract/qwen3_vl_2b_fp16_20260812.json
```

结果：命令返回 0，报告 SHA-256 为：

```text
48f3b27710bf9b0553bf3ef36f46f86e8bffcbd7368cd811e2cd7ee3f34cd162
```

关键字段：

```text
schema_version                         = parksight_prompt_contract_report_v1
workload_identity                      = parking_risk_v1@sha256:8350ace4574f8aa154319f7136ef831003d4dcc074ef20b74c1b419d69a2a493
message_contract.messages_equal        = true
message_contract.system_content_type   = array
message_contract.user_content_types    = ["image", "text"]
```

`transformers_rendered_prompt` 中存在 `events`、`evidence`、`driver_advice` 的 JSON 数组
约束和“不得把数组字段输出为普通字符串”要求。engine 使用的
`processed_chat_template.json` 为 633 bytes，SHA-256 为：

```text
58de245036cec9241ca19128288c80871ee287ebc4c489ce0dbdba991cfbb619
```

这证明在 Python Adapter 边界，Transformers 与 Edge HTTP 请求的结构化 messages 已经
对齐；它仍不等价于 Edge C++ Runtime 已经输出有效 JSON。

### 23.4 使用原 FP16 engine 启动服务

启动命令：

```bash
nohup env \
  PYTHONPATH=/home/ubuntu/TensorRT-Edge-LLM:/home/ubuntu/JetsonVLM/src \
  BUILD_DIR=/home/ubuntu/TensorRT-Edge-LLM/build \
  EDGELLM_PLUGIN_PATH=/home/ubuntu/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so \
  EDGELLM_WEIGHT_STREAMING_BUDGET_BYTES=0 \
  LD_LIBRARY_PATH=/home/ubuntu/JetsonVLM/.venv-jetson/lib/python3.10/site-packages/nvidia/cu12/lib:/home/ubuntu/TensorRT-Edge-LLM/build:/usr/local/cuda/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu \
  /home/ubuntu/JetsonVLM/.venv-jetson/bin/python \
  /home/ubuntu/JetsonVLM/scripts/serve_edgellm.py \
  --engine-root /home/ubuntu/JetsonVLM/artifacts/engines/qwen3_vl_2b_fp16 \
  --weight-streaming-budget-bytes 0 \
  --host 127.0.0.1 \
  --port 8000 \
  > /home/ubuntu/JetsonVLM/reports/prompt-contract/edgellm_server_system_array_20260812.log \
  2>&1 < /dev/null &
```

加载顺序实际到达：

```text
LLM engine 3293 MiB loaded
LLM Runtime tensors successfully allocated
tokenizer/chat template loaded
visual engine 785 MiB loaded
Vision runner successfully initialized
```

随后在共享执行上下文分配阶段失败：

```text
NvMapMemAllocInternalTagged: ... error 12
RuntimeError: CUDA runtime error in cudaMalloc(&data, memoryCapacity): out of memory
health_http=000
```

`visual/action/action.engine` 缺失只触发可选 Action runner 的捕获异常；运行时代码在该
异常之后继续执行，真正终止服务的是共享上下文的 `cudaMalloc` OOM。

同一批原 engine 在 2026-08-10 的既有日志中曾成功启动，关键证据为：

```text
base required context   = 1356027904 bytes
vision required context = 423624704 bytes
Application startup complete
Uvicorn running on http://127.0.0.1:8000
```

因此本次结果不能解释为 engine 与 Jetson 不兼容，而应继续检查本次启动前后的统一内存
连续块状态。

### 23.5 统一内存与连续块检查

模型进程退出后的状态：

```text
RAM used/total = 1493/7619 MiB
MemAvailable   = 约 6.0 GiB
lfb            = 77 x 4 MiB
Swap           = 约 450 MiB / 12002 MiB
GR3D_FREQ       = 0%
```

虽然 `free` 显示约 6 GiB 可用，但 `tegrastats` 的最大连续空闲块只有约 308 MiB。该状态
与“大量总可用内存仍存在，但 NvMap/CUDA 大块分配失败”的现象一致。板端已连续运行约两天，
且当时没有残留 Edge 服务进程。

还执行了以下非破坏性缓存回收实验：

```text
sync
对 llm.engine、visual.engine、embedding.safetensors 调用 POSIX_FADV_DONTNEED
```

页缓存下降，但 `lfb` 只从 `77 x 4 MiB` 增至 `85 x 4 MiB`，不足以解决共享上下文分配。
尝试通过 SSH 用户执行 `systemctl reboot` 时，系统明确返回：

```text
Interactive authentication required.
```

因此 Codex 无法在无 `sudo` 凭据的 SSH 会话内完成内核级内存整理或重启。

### 23.6 缩小视觉 profile 的排除实验

为确认 OOM 是否由视觉 profile 引起，保留原始 `maxImageTokens=2048` engine，并在新目录
构建一个不覆盖原产物的 `maxImageTokens=1024` 版本：

```bash
cd /home/ubuntu/JetsonVLM

PYTHONPATH=src .venv-jetson/bin/python scripts/build_edgellm_vlm_engines.py \
  --edge-llm-root /home/ubuntu/TensorRT-Edge-LLM \
  --expected-revision 7f061f21f0a581ba234a1e233c9315b89d8e47d6 \
  --onnx-root artifacts/onnx/qwen3_vl_2b_fp16 \
  --engine-root artifacts/engines/qwen3_vl_2b_fp16_img1024 \
  --component visual \
  --workspace-limit-mib 1024 \
  --builder-optimization-level 1 \
  --min-image-tokens 8 \
  --max-image-tokens 1024 \
  --max-image-tokens-per-image 1024
```

构建成功：

```text
TensorRT engine generation = 28.1423 s
Activation Memory          = 211812352 bytes
TRT GPU peak               = 789 MiB
TRT build CPU peak         = 4163 MiB
visual.engine bytes        = 823720316
visual.engine SHA-256      = 2312d0de16f8d4b61706ce05fc0964410618f35891c73bed55b6af8cbc2b0f33
```

新 engine 目录通过符号链接复用原 LLM engine：

```text
artifacts/engines/qwen3_vl_2b_fp16_img1024/llm
  -> ../qwen3_vl_2b_fp16/llm
```

使用新视觉 engine 重启服务后，仍在 `Vision runner successfully initialized` 之后的同一
共享执行上下文 `cudaMalloc` 位置 OOM。由此排除“2048-image-token 视觉 profile 是本次
启动失败的直接原因”；当前共享内存峰值由 LLM base context 主导。

### 23.7 本机保存的证据

本次远端证据已复制到：

```text
reports/jetson-runtime-20260812/qwen3_vl_2b_fp16_20260812.json
reports/jetson-runtime-20260812/edgellm_server_system_array_20260812.log
reports/jetson-runtime-20260812/build_visual_img1024_20260812.log
reports/jetson-runtime-20260812/edgellm_server_img1024_system_array_20260812.log
reports/jetson-runtime-20260812/visual_img1024_config.json
reports/jetson-runtime-20260812/jetson_state_20260812.txt
```

这些文件均位于被 Git 忽略的 `reports/` 目录，用于保留本地实验事实，不进入源码提交。

### 23.8 当前结论与恢复后的验收顺序

当前可以确认：

- Prompt 契约报告已在真实 Jetson processor、真实 workload 和真实 engine chat template
  上通过；
- 原 FP16 engine 和缩小视觉 profile 的 engine 均在本次启动中因统一内存大块分配失败；
- 本次尚未到达 HTTP 请求阶段，因此不能判断 system content 数组修改是否解决 0/20
  JSON 失败；
- 不应继续重复启动，以免进一步加剧内存碎片。

需要在 Jetson 本机执行一次：

```bash
sudo reboot
```

重启后首先检查 `free -h`、`swapon --show` 和 `tegrastats` 的 `lfb`，再使用原始
`qwen3_vl_2b_fp16` engine 启动服务。健康接口返回 200 后，执行 23.9 所列单图命令；
首轮严格 JSON 成功才继续到 3/3、5/5 和 20 样本 Study。

### 23.9 重启后待执行的单图命令

```bash
cd /home/ubuntu/JetsonVLM

PYTHONPATH=src .venv-jetson/bin/python -m parksight_vlm.app.analyze_image \
  --image data/raw/ps2.0/pilot/indoor/001.jpg \
  --workload configs/workloads/parking_risk_v1.json \
  --runtime tensorrt_edge_llm_http \
  --backend-revision 7f061f21f0a581ba234a1e233c9315b89d8e47d6 \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --model-revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --adapter-revision edge-http-system-content-array-v1 \
  --precision fp16 \
  --edge-url http://127.0.0.1:8000
```

### 23.10 2026-08-12 重启后的第一次恢复尝试

Jetson 重启约 5 分钟后重新检查：

```text
Mem total/used/available = 7.4 GiB / 1.3 GiB / 5.9 GiB
zram swap                = 3.7 GiB，used 0
8 GiB file swap          = 未启用
Edge-LLM process         = 无
health_http              = 000
```

`/home/ubuntu/parksight-build.swap` 文件仍存在，大小为 `8589934592` bytes，但它没有
配置进 `/etc/fstab`，因此不会随重启自动启用。文件当时的属主为 `ubuntu:ubuntu`、权限
为 `0600`。SSH 会话执行 `sudo -n swapon` 返回 `sudo: a password is required`。

在只有 zram 的状态下，使用 23.4 完全相同的原始 engine 启动命令做了一次受控重试。
LLM engine、Runtime tensors、tokenizer 和 visual engine 均加载完成，但仍在
`Vision runner successfully initialized` 之后分配共享执行上下文时失败：

```text
NvMapMemAllocInternalTagged: ... error 12
RuntimeError: CUDA runtime error in cudaMalloc(&data, memoryCapacity): out of memory
health_http=000
```

失败后的状态为：

```text
RAM used/total = 2385/7619 MiB
lfb            = 17 x 4 MiB
zram used      = 42/3810 MiB
服务进程       = 已退出
```

对应日志已保存为：

```text
reports/jetson-runtime-20260812/edgellm_server_after_reboot_20260812.log
```

因此“只重启、不重新启用 8 GiB 文件 swap”仍不足以恢复服务。下一次尝试前必须先在
Jetson 的交互终端执行：

```bash
sudo chown root:root /home/ubuntu/parksight-build.swap
sudo chmod 600 /home/ubuntu/parksight-build.swap
sudo swapon --priority 1 /home/ubuntu/parksight-build.swap
swapon --show
```

成功标准是 `swapon --show` 同时出现六个 `/dev/zram*` 和
`/home/ubuntu/parksight-build.swap`，总 swap 约为 11 GiB。启用后再启动一次原始
`qwen3_vl_2b_fp16` engine；在此之前不继续重复加载。

### 23.11 文件 swap 启用后的第二次恢复尝试

人工输入 `sudo` 密码后，文件 swap 已成功启用：

```text
/dev/zram0 ... /dev/zram5       total 约 3.7 GiB，priority 5
/home/ubuntu/parksight-build.swap       8 GiB，priority 1
总 swap                               约 11 GiB
```

此时再次使用原 engine 启动，仍在相同位置 OOM。失败后的证据：

```text
MemAvailable = 约 5.0 GiB
Swap used    = 约 102 MiB / 11 GiB
lfb          = 30 x 4 MiB
服务进程     = 已退出
```

既有成功日志明确记录共享执行上下文分配为：

```text
shared context = 1356027904 bytes
base requires  = 1356027904 bytes
vision requires = 423624704 bytes
```

本次最大连续空闲块约为 120 MiB，而服务需要创建约 1.26 GiB 的共享上下文。文件 swap
可以换出普通 CPU 页面，却不会自动整理已经碎片化的物理内存，也不能直接作为
`cudaMalloc` 的 GPU buffer。因此，先在无文件 swap 状态下失败一次、再启用 swap，无法
恢复第一次加载已经破坏的连续内存布局。

对应日志保存为：

```text
reports/jetson-runtime-20260812/edgellm_server_swap_enabled_20260812.log
```

下一步不重复加载模型，而是先执行一次内核页缓存回收和主动内存整理：

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory'
```

随后用 `free -h` 和 `tegrastats` 复查。若整理后仍无法启动，则将 swap 写入 `/etc/fstab`
并再次重启，使 8 GiB 文件 swap 在第一次 engine 加载之前就处于启用状态。

### 23.12 实际输入规模测量与 640-token 诊断 engine

仅根据 workload 中的 `max_new_tokens=256` 无法确定 engine 的输入 profile，因为图片经过
processor 和 Edge-LLM 模板展开后还会引入视觉占位 token。为此，使用真实模型
processor、真实 workload、真实图片和已导出的 chat template 测量输入：

```bash
cd /home/ubuntu/JetsonVLM
PYTHONPATH=src .venv-jetson/bin/python \
  reports/prompt-contract-20260811/measure_qwen_input.py
```

Python processor 的直接结果为：

```text
input_ids shape             = [1, 570]
visual grid                 = [1, 28, 28]
visual tokens after merge   = 196
max_new_tokens              = 256
estimated minimum KV length = 826
```

测量证据保存为：

```text
reports/jetson-runtime-20260812/qwen_input_measurement_20260812.json
```

随后构建 `maxInputLen=640`、`maxKVCacheCapacity=1024` 的诊断 engine，用于验证缩小
profile 是否能够降低共享执行上下文：

```bash
cd /home/ubuntu/JetsonVLM
PYTHONPATH=src .venv-jetson/bin/python scripts/build_engine.py \
  --config configs/flows/build_qwen3_vl_2b_fp16_llm_engine_i640_k1024.json \
  --execute
```

该诊断 engine 的关键结果为：

```text
engine path = artifacts/engines/qwen3_vl_2b_fp16_i640_k1024/llm/llm.engine
engine bytes = 3453811124
SHA-256      = 6e9b93c5ca01ed920147b3a079cf8be8fb2e39c424f5c3cf8cfa8de4b239dc3c
build time   = 143.87 s
TRT CPU/GPU allocator peak = 2578/703 MiB
build+serialization CPU peak = 6266 MiB
```

在后续 C++ 分配顺序补丁生效后，该 engine 可以启动并通过 `/health`，但真实 HTTP 请求
返回 500：Edge-LLM C++ 侧最终展开输入为 735 token，超过 640 上限。该结果说明 Python
processor 的 570 token 只能用于预估，最终 profile 必须以实际运行时模板展开长度为准。
因此 640 配置只保留为诊断证据，不作为最终运行配置。

### 23.13 TensorRT weight streaming 预算探测

使用 JetPack 系统 Python 中的 TensorRT 10.3 API读取 LLM engine 的 weight streaming
属性，命令为：

```bash
cd /home/ubuntu/JetsonVLM
/usr/bin/python3 reports/prompt-contract-20260811/probe_weight_streaming.py \
  artifacts/engines/qwen3_vl_2b_fp16_i640_k1024/llm/llm.engine
```

探测结果为：

```text
streamable weights              = 3441150208 bytes
budget=0 scratch                = 1244660224 bytes
budget=256 MiB ... 3 GiB        = scratch 不下降，且总占用随 budget 增长
budget=3441150208 bytes          = scratch 归零，但直接常驻权重申请 OOM
```

因此当前 8 GB Jetson 上 `--weight-streaming-budget-bytes 0` 是实测总内存占用最低的选择；
中间预算会同时保留 streaming budget 和约 1.16 GiB scratch，反而更差；全量 budget 则
无法分配约 3.20 GiB 连续驻留权重。完整输出保存为：

```text
reports/jetson-runtime-20260812/weight_streaming_probe_20260812.json
```

### 23.14 OOM 根因与 TensorRT Edge-LLM C++ 补丁

对成功与失败日志的加载顺序进行对比后，定位到直接根因：原始 runtime 先加载约
785 MiB 的 visual engine 权重，再一次性申请约 1.35 GB 的 LLM 共享 execution context。
在 Jetson 统一内存已经被模型、embedding、weight-streaming scratch 和系统服务占用后，
这一大块 `cudaMalloc` 容易失败。文件 swap 只缓解可换出的 CPU 页面，不能代替 CUDA
buffer，也不能保证形成所需的连续设备映射。

修复方式是在加载可选 multimodal runner 之前：

1. 查询 base executor 和 decoder strategy 的 context 大小；
2. 先申请并绑定共享 GPU context；
3. 再加载 visual/audio/action runner；
4. 只有 multimodal runner 的需求确实更大时才重新分配。

补丁已固化为：

```text
patches/tensorrt-edge-llm/preallocate-base-context-before-multimodal.patch
```

在一份干净、固定于 `7f061f21f0a581ba234a1e233c9315b89d8e47d6` 的
TensorRT Edge-LLM 源码上，复现命令为：

```bash
cd /home/ubuntu/TensorRT-Edge-LLM
git apply --check \
  /home/ubuntu/JetsonVLM/patches/tensorrt-edge-llm/preallocate-base-context-before-multimodal.patch
git apply \
  /home/ubuntu/JetsonVLM/patches/tensorrt-edge-llm/preallocate-base-context-before-multimodal.patch

/home/ubuntu/JetsonVLM/.venv-jetson/bin/cmake --build build \
  --target _edgellm_runtime -j2
```

本次增量编译成功，产物为：

```text
/home/ubuntu/TensorRT-Edge-LLM/build/pybind/_edgellm_runtime.cpython-310-aarch64-linux-gnu.so
size    = 47559536 bytes
SHA-256 = cc9518f4adbe9a6e2cf7512b9cf9803e80d6491d09bf8d8a46092b966d2ab4de
source patch SHA-256 = 3570e0bb61c281b9f74842db2526cb22ebc32a3f4b9aa41c69920deed7a8de43
```

640/1024 engine 启动日志首次明确出现：

```text
Preallocated base execution context memory: 1343444992 bytes
```

并在之后才加载 visual engine，`/health` 返回 200，证明分配顺序修改解决了该 profile 的
启动 OOM。原始 1024/2048 engine 虽然也能先申请 1,356,027,904 字节 context，但随后
加载视觉权重仍因总峰值过高而 OOM，因此还必须缩小 engine profile。

### 23.15 最终 768/1024 FP16 engine

真实 C++ 请求长度为 735 token，因此最终 profile 选择：

```text
max batch size       = 1
max input length     = 768
max KV cache capacity = 1024
workspace limit      = 1024 MiB
builder optimization = 0
weight streaming     = enabled
runtime budget       = 0 bytes
```

仓库中的可复现配置为：

```text
configs/flows/build_qwen3_vl_2b_fp16_llm_engine_i768_k1024.json
```

构建命令：

```bash
cd /home/ubuntu/JetsonVLM
PYTHONPATH=src .venv-jetson/bin/python scripts/build_engine.py \
  --config configs/flows/build_qwen3_vl_2b_fp16_llm_engine_i768_k1024.json \
  --execute

ln -s ../qwen3_vl_2b_fp16/visual \
  artifacts/engines/qwen3_vl_2b_fp16_i768_k1024/visual
```

构建实测结果：

```text
engine path = artifacts/engines/qwen3_vl_2b_fp16_i768_k1024/llm/llm.engine
engine bytes = 3453786212
SHA-256      = 5a679dceb7fdbe5661dd5ae67ba28804f0e18341f10adeb99b2906c2d8cfcf05
engine generation time = 148.656 s
activation memory       = 102979072 bytes
TRT CPU/GPU allocator peak = 2629/652 MiB
build+serialization CPU peak = 6725 MiB
```

构建期间 8 GiB 文件 swap 与 zram 均启用；该 swap 是构建和 CPU 内存压力的保险措施，
不是 runtime `cudaMalloc` OOM 的直接修复。

### 23.16 最终 server 启动与单图 3/3 验收

最终 server 使用补丁后的 `_edgellm_runtime`、768/1024 LLM engine 和已成功构建的
visual engine：

```bash
cd /home/ubuntu/JetsonVLM
export EDGE_LLM_ROOT=/home/ubuntu/TensorRT-Edge-LLM
export PYTHONPATH=$EDGE_LLM_ROOT:$PWD/src
export EDGELLM_PLUGIN_PATH=$EDGE_LLM_ROOT/build/libNvInfer_edgellm_plugin.so
export JETSON_PY_CUDA_LIB=$PWD/.venv-jetson/lib/python3.10/site-packages/nvidia/cu12/lib
export LD_LIBRARY_PATH=$JETSON_PY_CUDA_LIB:$EDGE_LLM_ROOT/build:$LD_LIBRARY_PATH

.venv-jetson/bin/python scripts/serve_edgellm.py \
  --engine-root artifacts/engines/qwen3_vl_2b_fp16_i768_k1024 \
  --weight-streaming-budget-bytes 0 \
  --host 127.0.0.1 \
  --port 8000
```

启动成功证据：

```text
Preallocated base execution context memory: 1347639296 bytes
Setup shared execution context memory: 1347639296 bytes
vision requires: 423624704 bytes
Application startup complete
GET /health = HTTP 200
```

使用冻结 workload 和 PS2.0 pilot 同一图片连续运行三次：

```bash
cd /home/ubuntu/JetsonVLM
PYTHONPATH=src .venv-jetson/bin/python -m parksight_vlm.app.analyze_image \
  --image data/raw/ps2.0/pilot/indoor/001.jpg \
  --workload configs/workloads/parking_risk_v1.json \
  --runtime tensorrt_edge_llm_http \
  --backend-revision 7f061f21f0a581ba234a1e233c9315b89d8e47d6 \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --model-revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --adapter-revision edge-http-system-content-array-v1 \
  --precision fp16 \
  --edge-url http://127.0.0.1:8000
```

三次结果均满足：HTTP 200、`failure=null`、`assessment` 非空、数组字段类型正确、所有
枚举值通过严格 schema。三次结构化结果完全一致：

```json
{
  "risk_level": "low",
  "events": ["narrow_passage", "vehicle_near_maneuver_path"],
  "driver_advice": ["maintain_observation", "slow_down", "yield"]
}
```

其中 `evidence` 为 3 条字符串，完整输出保存在对应 JSON 证据中。性能结果：

| 次数 | 端到端时延 | 输出 token | 生成速度 |
| --- | ---: | ---: | ---: |
| 1 | 76272.771 ms | 114 | 1.495 token/s |
| 2 | 76158.795 ms | 114 | 1.497 token/s |
| 3 | 76138.652 ms | 114 | 1.497 token/s |
| 平均 | 76190.073 ms | 114 | 1.496 token/s |

遥测共 510 个采样点：

```text
peak RAM       = 7422 MiB
peak swap      = 1550 MiB
peak GPU util  = 99%
average GPU util = 87.49%
peak GPU temp  = 62.375 C
peak power     = 10.56 W
average power  = 9.47 W
```

workload 身份为：

```text
parking_risk_v1@sha256:8350ace4574f8aa154319f7136ef831003d4dcc074ef20b74c1b419d69a2a493
```

本轮结束后 server 已正常停止。板端恢复到约 1.6 GiB RAM used、5.7 GiB available、
约 913 MiB swap used，且无残留 server 进程。主要证据文件为：

```text
reports/jetson-runtime-20260812/build_runtime_preallocate_context_20260812.log
reports/jetson-runtime-20260812/build_llm_i768_k1024_20260812.log
reports/jetson-runtime-20260812/edgellm_server_i768_k1024_preallocate_context_20260812.log
reports/jetson-runtime-20260812/single_i768_run1_20260812.json
reports/jetson-runtime-20260812/single_i768_run2_20260812.json
reports/jetson-runtime-20260812/single_i768_run3_20260812.json
reports/jetson-runtime-20260812/tegrastats_i768_run1_20260812.log
reports/jetson-runtime-20260812/tegrastats_i768_runs2_3_20260812.log
```

### 23.17 当前客观结论与下一验收层级

当前已经形成真实的 Jetson TensorRT Edge-LLM FP16 单图闭环：固定模型 revision 与
workload，ONNX/engine 构建，补丁后的 runtime 启动，图片请求，模型生成，严格 JSON
解析以及功耗、温度、内存和时延采集均有实测证据。原先 0/20 的主要输出格式问题在该
冻结样本上已由 system content 数组契约修复，并通过连续 3/3 验收。

边界同样明确：3/3 只证明同一图片的稳定性，不能替代完整数据集质量结论；当前
FP16+weight streaming 速度约 1.50 token/s，仍属于正确性/部署基线，不是最终加速结果。
因此随后继续使用相同 768/1024 engine 运行冻结的 20 样本 Study，结果见 23.18。

### 23.18 冻结 20 样本最终 Study

新增独立 study 配置，避免覆盖 2026-08-10 的 0/20 历史失败报告：

```text
configs/studies/jetson_edgellm_fp16_ps20_pilot_promptfix.json
study_id = jetson_edgellm_fp16_ps20_pilot_promptfix_i768_k1024
adapter_revision = edge-http-system-content-array-v1
output = reports/jetson_edgellm_fp16_ps20_pilot_promptfix_i768_k1024.json
```

服务健康后，启动 2 秒周期的板端遥测并运行完整 Study：

```bash
cd /home/ubuntu/JetsonVLM

tegrastats --interval 2000 \
  > reports/tegrastats_promptfix_i768_k1024_20sample_20260812.log 2>&1 &

PYTHONPATH=src .venv-jetson/bin/python -m parksight_vlm.app.run_study \
  --config configs/studies/jetson_edgellm_fp16_ps20_pilot_promptfix.json
```

先观察前 5 个 HTTP 200 作为冒烟门槛，再保持同一 server、engine 和 workload 完成全部
20 个样本。运行过程中 RAM 长期稳定在约 7.34 GiB、swap 稳定在约 1.85 GiB，没有随
请求数持续增长，也没有再次触发 `cudaMalloc` OOM 或输入 profile 越界。

最终 StudyReport：

```text
sample count                       = 20
backend completed                  = 20/20
strict JSON valid                  = 20/20 (100%)
failure summary                    = {}
risk level accuracy                = 0.35
event micro precision/recall/F1    = 0.3043 / 0.4375 / 0.3590
unsafe advice rate                 = 0
end-to-end mean                    = 53.639 s
end-to-end p50/p90/p99             = 50.753 / 69.082 / 75.082 s
output tokens total/mean           = 1588 / 79.4
aggregate output token rate        = 1.480 token/s
```

相对于 2026-08-10 的同一冻结 20 样本，严格 JSON 有效率从 `0/20` 恢复到 `20/20`，
说明 system content 数组修复已经覆盖完整 pilot，而不仅是单张图片。质量指标也说明
当前瓶颈已经从“响应格式不可用”转为“领域判断偏差”：模型倾向预测 `narrow_passage`
（13 个 false positive），同时漏掉 9 个 `visibility_occlusion`；这构成 LoRA 数据选择与
错误分析的直接依据。

使用已有摘要入口处理 StudyReport 与 542 条 tegrastats：

```bash
PYTHONPATH=src python3 scripts/summarize_jetson_study.py \
  --study-report reports/jetson_edgellm_fp16_ps20_pilot_promptfix_i768_k1024.json \
  --tegrastats reports/tegrastats_promptfix_i768_k1024_20sample_20260812.log \
  --output reports/jetson_edgellm_fp16_ps20_pilot_promptfix_i768_k1024_summary.json
```

遥测结果：

```text
sample count          = 542
RAM mean/peak         = 7410.73 / 7418 MB
swap mean/peak        = 1856.57 / 1904 MB
GPU utilization mean = 97.39%, peak 99%
GPU temperature mean/peak = 63.13 / 65.03 C
VDD_IN mean/peak      = 10.05 / 10.70 W
```

运行结束后停止 server 和 tegrastats。板端 RAM 回落到约 1.68 GiB、swap 回落到约
908 MiB。报告、server 日志、遥测日志和派生摘要均已同步到本机忽略目录：

```text
reports/jetson-runtime-20260812/
  jetson_edgellm_fp16_ps20_pilot_promptfix_i768_k1024.json
  jetson_edgellm_fp16_ps20_pilot_promptfix_i768_k1024_summary.json
  edgellm_server_promptfix_20sample_20260812.log
  tegrastats_promptfix_i768_k1024_20sample_20260812.log
  study_promptfix_i768_k1024_20sample_20260812.stdout.log
```

至此，当前 FP16 部署阶段的目标已经完成：固定版本、可复现构建、Jetson runtime、严格
结构化输出、20 样本质量评测和板端性能/功耗证据全部闭环。后续工作不再是继续修这条
链路，而是用独立 train/validation 数据训练 LoRA，以及构建 INT4 对照；二者必须保留
本节的冻结 test 集和 768/1024 FP16 结果作为对照基线。

## 24. INT4 阶段启动记录（2026-08-12）

### 24.1 FP16 阶段提交与推送

在进入 INT4 前，先提交已经通过 Jetson 20 样本验收的 FP16 修复：

```bash
git add -- \
  docs/edgellm-deployment.md \
  docs/execution-report.md \
  docs/progress.md \
  docs/status.md \
  src/parksight_vlm/inference/edge_llm.py \
  src/parksight_vlm/inference/transformers.py \
  src/parksight_vlm/inference/prompt_contract.py \
  scripts/inspect_prompt_contract.py \
  tests/inference/test_runtime.py \
  tests/inference/test_prompt_contract.py \
  tests/fixtures/prompt_contract/model/.gitkeep \
  tests/fixtures/prompt_contract/processed_chat_template.json \
  configs/flows/build_qwen3_vl_2b_fp16_llm_engine_i768_k1024.json \
  configs/studies/jetson_edgellm_fp16_ps20_pilot_promptfix.json \
  patches/tensorrt-edge-llm/preallocate-base-context-before-multimodal.patch

git commit -m "fix: stabilize Jetson Edge-LLM FP16 runtime"
git push
```

结果：

```text
commit = 7c1567c
branch = main
remote = origin/main
push   = 6eb178b..7c1567c
```

个人文件 `1.md`、`docs/record.md` 和 `reports/` 原始运行证据没有加入提交；前者属于用户
未跟踪文件，后两者由 `.gitignore` 明确作为个人记录和本地大体积证据保留。

### 24.2 TensorRT Edge-LLM INT4 工作流确认

固定 revision `7f061f21f0a581ba234a1e233c9315b89d8e47d6` 的源码与文档确认：

```text
FP16/BF16 checkpoint
  -> tensorrt-edgellm-quantize（ModelOpt 校准与 checkpoint 量化）
  -> quantized safetensors + hf_quant_config.json
  -> tensorrt-edgellm-export（产生 INT4 custom-op ONNX）
  -> Jetson llm_build（集成 Int4GroupwiseGemmPlugin）
  -> TensorRT Edge-LLM runtime
```

关键边界：

- Jetson Orin + JetPack 6.2 支持 FP16、INT8 和 INT4；不支持 FP8、NVFP4 等 runtime
  精度；
- INT4 groupwise plugin 使用对称 weight-only 量化，当前 group size 为 128；
- Qwen3-VL 的目标组合为 LLM backbone INT4 AWQ、视觉编码器 FP16；
- 不指定 `--visual_quantization` 时视觉塔保持 FP16；
- 不指定 `--lm_head_quantization` 时 LM head 保持 FP16；
- 量化和 ONNX 导出必须在 x86-64 Linux + NVIDIA GPU 上执行，Jetson 只承担 C++
  runtime 与 engine 构建；
- 2B 模型官方建议至少 8--16 GB VRAM，因而优先使用既有 AutoDL GPU 实例。

预定量化命令采用 128 条 `cnn_dailymail` 文本样本，在项目时间优先的约束下缩短校准
时间，同时将样本数写入产物名称，避免与默认 512 样本结果混淆：

```bash
cd /root/JetsonVLM

PYTHONPATH=src python3 scripts/quantize_model.py \
  --config configs/flows/quantize_qwen3_vl_2b_int4_awq.json

PYTHONPATH=src python3 scripts/quantize_model.py \
  --config configs/flows/quantize_qwen3_vl_2b_int4_awq.json \
  --execute
```

等价的核心外部命令为：

```bash
tensorrt-edgellm-quantize llm \
  --model_dir models/Qwen3-VL-2B-Instruct-89644892 \
  --output_dir artifacts/quantized/qwen3_vl_2b_int4_awq_n128 \
  --quantization int4_awq \
  --text_dataset cnn_dailymail \
  --num_samples 128
```

### 24.3 已实现的 INT4 可复现入口

新增：

```text
scripts/quantize_model.py
configs/flows/quantize_qwen3_vl_2b_int4_awq.json
configs/flows/export_qwen3_vl_2b_int4_awq.json
configs/flows/build_qwen3_vl_2b_int4_awq_llm_engine_i768_k1024.json
configs/studies/jetson_edgellm_int4_awq_ps20_pilot.json
```

同时将 `quantize_model` 加入 `ExternalFlowPlan` 的合法阶段。最终 Jetson engine 继续使用
FP16 已验证的 `input=768`、`KV=1024` 口径，但不启用 weight streaming；INT4 权重应
显著降低常驻内存，是否确实无需 streaming 必须以板端启动证据为准。

本地验证：

```text
41 tests OK
21 JSON configs parsed successfully
Python compileall passed
git diff --check passed
```

### 24.4 旧实例失效与新实例恢复

原 GPU 服务器地址：

```text
ssh -p 41265 root@connect.westb.seetacloud.com
```

当前连接在 SSH banner 阶段直接返回 `Connection refused`，说明旧 AutoDL 实例未运行、
端口已变化或实例已释放。Windows 本机虽有 RTX 4060 Laptop 8 GB，但未安装 WSL；
固定版本 Edge-LLM 明确要求 x86-64 Linux GPU 导出环境，因此不在 Windows 上临时建立
一套不可比较的工具链。Jetson 也不是受支持的 checkpoint 量化/ONNX 导出主机。

随后重新租用 AutoDL RTX 4090 D 实例并恢复执行。新实例连接凭据仅用于当次 SSH，
密码未写入仓库、记录或脚本。实测环境如下：

```text
OS                 Ubuntu 22.04.5 LTS x86_64
GPU                NVIDIA GeForce RTX 4090 D, 24564 MiB
driver             595.91.07
base Python        /root/miniconda3/bin/python, 3.12.3
base PyTorch       2.8.0+cu128
system CUDA        12.8, nvcc 12.8.93
data disk          /root/autodl-tmp, 50 GiB
RAM                503 GiB
```

为避免 30 GiB 系统盘被模型与 wheel 占满，源码、venv、模型、缓存和产物均放到
`/root/autodl-tmp`。GitHub clone 在网络侧长时间无进展，因此将本地固定提交
`7c1567c` 归档上传；TensorRT Edge-LLM 同样从本地固定 commit 归档上传，服务端写入
`.source-commit` 记录来源，而不伪造 Git 历史。

### 24.5 新实例 INT4 工具环境

固定工具环境位于：

```text
/root/autodl-tmp/TensorRT-Edge-LLM/.venv-int4
```

核心创建命令：

```bash
/root/miniconda3/bin/python -m venv \
  /root/autodl-tmp/TensorRT-Edge-LLM/.venv-int4

/root/autodl-tmp/TensorRT-Edge-LLM/.venv-int4/bin/python -m pip install \
  torch==2.12.0 torchvision==0.27.0

/root/autodl-tmp/TensorRT-Edge-LLM/.venv-int4/bin/python -m pip install \
  -e '/root/autodl-tmp/TensorRT-Edge-LLM[tools]'
```

PyTorch 官方 `cu126` index 下载在该实例上极慢，保留缓存后切换到 AutoDL 配置的阿里云
PyPI 镜像。当前默认 `torch==2.12.0` wheel 实际为 CUDA 13.0 构建；服务器驱动支持，
CUDA smoke test 通过。最终固定版本与验证结果：

```text
TensorRT Edge-LLM  0.9.1 / 7f061f21f0a581ba234a1e233c9315b89d8e47d6
PyTorch            2.12.0, torch CUDA 13.0
torchvision        0.27.0
Transformers       5.9.0
ModelOpt           0.44.0
datasets           4.8.5
ONNX               1.19.0
ONNX Script        0.7.0
pip check          No broken requirements found
CUDA available     true, NVIDIA GeForce RTX 4090 D
```

这只是 checkpoint 量化与 ONNX 导出环境。Jetson engine 仍由 Jetson 上的 TensorRT 10.3
和 Edge-LLM C++ builder 构建，不能把服务器 CUDA 13 环境描述成板端 runtime 环境。

### 24.6 固定模型下载

Hugging Face 主站直连超时，镜像单连接过慢。先通过 `hf download` 获取 11 个小文件并
建立大权重断点，随后启用 AutoDL 学术网络加速并使用 `aria2c` 16 路 Range 下载权重。
平均下载速度约 18 MiB/s。最终严格校验：

```text
model revision = 89644892e4d85e24eaac8bacfd4f463576704203
weight bytes   = 4255140312
weight sha256  = 7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0
```

该长度与哈希均和此前 FP16 导出使用的固定 checkpoint 一致。

### 24.7 校准集兼容问题与固定样本

第一次量化在模型加载后、GPU 校准前失败：固定版 Edge-LLM 内置数据集使用旧短名
`cnn_dailymail`，当前 Hugging Face Hub 客户端要求 `namespace/name`，报错为：

```text
HfUriError: Repository id must be 'namespace/name', got 'cnn_dailymail'
```

未修改上游源码，而是使用 Edge-LLM 已实现的环境变量 override：从 Hugging Face Dataset
Viewer API 读取 `abisee/cnn_dailymail` / `3.0.0` / `train` 的前 128 条文章，按 canonical
JSONL 保存，并设置：

```bash
export EDGELLM_QUANT_DATASET_CNN_DAILYMAIL=\
/root/autodl-tmp/calibration/cnn_dailymail_train_first128.jsonl
```

校准集身份：

```text
rows   = 128
bytes  = 491478
sha256 = bd8fcac1b7d052babd992f7365fd96bdfbfc785613a36d79ae1648c11d79ca97
```

失败 flow 的 JSON 和日志保留为 `failed_dataset_alias` 证据；正式 flow 使用同一个
`--text_dataset cnn_dailymail --num_samples 128` 命令成功。

### 24.8 INT4 AWQ 量化实测

正式命令：

```bash
cd /root/autodl-tmp/JetsonVLM
export PATH=/root/autodl-tmp/TensorRT-Edge-LLM/.venv-int4/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/JetsonVLM/src

python scripts/quantize_model.py \
  --config configs/flows/quantize_qwen3_vl_2b_int4_awq.json \
  --execute
```

结果：

```text
flow status                succeeded
flow elapsed               113.50 s
ModelOpt quantization      98.4 s
total tool time            103.7 s
peak observed GPU memory   about 8.5 GiB
quantized checkpoint bytes 2185741496
checkpoint sha256          4d6ac3707d68d084ae1580980d9c3fcbbf6dd956b6a6f1d5bd533bfd521c263d
hf_quant_config sha256     b110ce75c23ec080732624ecd32a45076a6c4897543116d484a2cec8a93b6d21
```

`hf_quant_config.json` 的实际语义：

```text
quant_algo       W4A16_AWQ
group_size       128
zero_point       false
pre_quant_scale  true
KV cache quant   none
excluded         lm_head, model.visual*
```

因此本实验只量化 LLM backbone；LM head、视觉塔和 KV cache 不量化。

### 24.9 INT4 ONNX 导出实测

```bash
python scripts/export_model.py \
  --config configs/flows/export_qwen3_vl_2b_int4_awq.json \
  --execute
```

结果为 `succeeded`，flow 用时 98.19 秒。LLM 主要文件：

```text
llm/model.onnx              4209515 bytes
llm/model.onnx.data         1350303744 bytes
llm/embedding.safetensors   622329944 bytes
```

对应 SHA-256：

```text
model.onnx             78c7badb4cfa1f81dd71a66742b811c89131e232c7af6522fd73956ea59e9603
model.onnx.data        235606ea014b1377560e6cc8ea22792a7a65c0289f5edfdf9d546e4911720724
embedding.safetensors  af9dbce9b96ca45a1cdc7c5ebdd18dc780567e707388e7cead78fbf1adb256d5
```

导出也生成 FP16 visual ONNX；由于模型 revision 相同且板端已有验证通过的 FP16 visual
engine，本次仅归档 LLM 目录，后续复用原 visual engine。LLM 归档：

```text
bytes  = 1988280320
sha256 = 346d367316bf1a03a13e03287419ce70265408cf54ecf6d736a594232e0e2986
```

### 24.10 服务器到 Jetson 的 INT4 ONNX 传输

服务器向 Windows 的单 SFTP 连接过慢，因此改为服务器回环 HTTP Range 服务，通过
Jetson 到服务器的 SSH local forwarding 暴露给板端本机，再由板端 `aria2c` 16 路断点
下载。服务端只监听 `127.0.0.1`，不把文件暴露到公网；密码仅在当次进程环境中传递，
没有写入仓库、记录或脚本。

核心执行形式如下，其中连接参数和凭据省略：

```bash
# AutoDL：仅监听回环地址的 Range HTTP 服务
python range_http_server.py \
  --bind 127.0.0.1 --port 18080 \
  --directory /root/autodl-tmp

# Jetson：建立本机端口到 AutoDL 回环端口的 SSH 隧道
ssh -N -L 127.0.0.1:18080:127.0.0.1:18080 <AUTODL_SSH_TARGET>

# Jetson：多连接、可断点续传下载
aria2c -c -x 16 -s 16 -k 1M \
  http://127.0.0.1:18080/qwen3_vl_2b_int4_awq_n128_llm.tar
```

下载平均速度约 3.7 MiB/s。板端归档路径与校验结果：

```text
/home/ubuntu/JetsonVLM/artifacts/transfers/
  qwen3_vl_2b_int4_awq_n128_llm.tar

archive sha256 = 346d367316bf1a03a13e03287419ce70265408cf54ecf6d736a594232e0e2986
```

解包到：

```text
/home/ubuntu/JetsonVLM/artifacts/onnx/qwen3_vl_2b_int4_awq_n128/llm
```

归档总 SHA-256 和 `llm.sha256` 中 7 个文件的逐文件校验全部通过。传输完成后已停止临时
HTTP 服务和 SSH 隧道。

### 24.11 Jetson INT4 LLM engine 构建

构建命令：

```bash
cd /home/ubuntu/JetsonVLM
export PYTHONPATH=/home/ubuntu/JetsonVLM/src

.venv-jetson/bin/python scripts/build_engine.py \
  --config configs/flows/build_qwen3_vl_2b_int4_awq_llm_engine_i768_k1024.json \
  --execute
```

实际结果：

```text
flow status             succeeded
flow elapsed            40.07 s
TensorRT engine build   28.62 s
TensorRT GPU peak       1617 MiB
build CPU peak          5693 MiB
max batch size          1
max input length        768
max KV cache length     1024
```

LLM engine 身份：

```text
path   artifacts/engines/qwen3_vl_2b_int4_awq_i768_k1024/llm/llm.engine
bytes  1362769140
sha256 33466f3f1149801bf496737fbe1e69634b8018ece9a0c5bccd8ff83af64afab4
```

visual engine 复用同一模型 revision 的 FP16 产物，通过 `visual` 符号链接组合为完整 engine
root；视觉 engine SHA-256 仍为：

```text
3c6b4cceaa946da54149587726a81546de8dff7183c6f74ddf09535ec005f07e
```

INT4 LLM engine 不需要 weight streaming。与 FP16 LLM engine 的 3453786212 bytes 相比，
文件缩小 60.5%。

### 24.12 INT4 plugin 加载问题与修复

首次启动 INT4 runtime 失败，日志显示 `Int4GroupwiseGemmPlugin` 未注册。原因不是 engine
损坏，而是启动目录不是 Edge-LLM 源码根目录，runtime 默认的相对路径没有找到已编译
plugin。显式固定 plugin 路径后启动成功：

```bash
cd /home/ubuntu/JetsonVLM
export EDGELLM_PLUGIN_PATH=/home/ubuntu/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
export LD_LIBRARY_PATH=/home/ubuntu/JetsonVLM/.venv-jetson/lib/python3.10/site-packages/nvidia/cu12/lib:/home/ubuntu/TensorRT-Edge-LLM/build:/usr/local/cuda/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu
export PYTHONPATH=/home/ubuntu/JetsonVLM/src:/home/ubuntu/TensorRT-Edge-LLM

.venv-jetson/bin/python scripts/serve_edgellm.py \
  --engine-root artifacts/engines/qwen3_vl_2b_int4_awq_i768_k1024 \
  --host 127.0.0.1 --port 8000
```

健康检查返回：

```json
{"status":"healthy","model":"/home/ubuntu/JetsonVLM/artifacts/engines/qwen3_vl_2b_int4_awq_i768_k1024/llm","speculative_decoding":false}
```

启动日志中的预分配量：

```text
LLM engine loaded size       1299 MiB
LLM base context prealloc    113995264 bytes
visual context prealloc      423624704 bytes
server RSS                   about 4.38 GiB
```

FP16 LLM base context 预分配为 1347639296 bytes；INT4 版本显著降低了板端上下文常驻内存。

### 24.13 串行单图稳定性验证

INT4 engine 的 `max_batch_size=1`。一次探索性测试错误地并发发送了两个请求，一个返回
`First dimension must match oldActiveBatch`，另一个生成损坏 JSON；该结果是超出 engine
并发契约的测试错误，不计入性能或质量结果。重启服务后严格串行执行 3 次：

```bash
for run in 1 2 3; do
  PYTHONPATH=src .venv-jetson/bin/python -m parksight_vlm.app.analyze_image \
    --image data/raw/ps2.0/pilot/indoor/001.jpg \
    --workload configs/workloads/parking_risk_v1.json \
    --runtime tensorrt_edge_llm_http \
    --backend-revision 7f061f21f0a581ba234a1e233c9315b89d8e47d6 \
    --model-id Qwen/Qwen3-VL-2B-Instruct \
    --model-revision 89644892e4d85e24eaac8bacfd4f463576704203 \
    --adapter-revision edge-http-system-content-array-v1 \
    --precision int4_awq \
    --edge-url http://127.0.0.1:8000 \
    > "reports/int4_smoke_serial_run${run}_20260813.json"
done
```

结果：

| run | 严格 JSON | 端到端时延 | 输出 tokens | 端到端 tokens/s |
| --- | --- | ---: | ---: | ---: |
| 1 | 通过 | 12779.85 ms | 94 | 7.36 |
| 2 | 通过 | 12695.24 ms | 94 | 7.40 |
| 3 | 通过 | 12686.39 ms | 94 | 7.41 |

3 次评估内容一致，证明单请求串行路径可重复；这不代表 runtime 已支持并发请求。

### 24.14 冻结 20 样本 INT4 Study

执行命令：

```bash
cd /home/ubuntu/JetsonVLM
tegrastats --interval 2000 \
  > reports/tegrastats_int4_awq_20sample_20260813.log &

PYTHONPATH=src .venv-jetson/bin/python -m parksight_vlm.app.run_study \
  --config configs/studies/jetson_edgellm_int4_awq_ps20_pilot.json

PYTHONPATH=src .venv-jetson/bin/python scripts/summarize_jetson_study.py \
  --study-report reports/jetson_edgellm_int4_awq_ps20_pilot_i768_k1024_n128.json \
  --tegrastats reports/tegrastats_int4_awq_20sample_20260813.log \
  --output reports/jetson_edgellm_int4_awq_ps20_pilot_i768_k1024_n128_summary.json
```

运行完整性：

```text
records                    20
backend completed          20/20
strict JSON valid          20/20
failure summary            empty
aggregate output rate      7.3276 tokens/s
```

性能与资源：

| 指标 | INT4 AWQ 实测 |
| --- | ---: |
| 端到端 mean | 10685.66 ms |
| 端到端 p50 | 10524.38 ms |
| 端到端 p90 | 11394.11 ms |
| 端到端 p99 | 12626.09 ms |
| RAM mean / peak | 5064.53 / 5072 MiB |
| swap mean / peak | 840 / 840 MiB |
| GPU 利用率 mean / peak | 98.28% / 99% |
| GPU 温度 mean / peak | 58.56 / 60.75 C |
| VDD_IN mean / peak | 10.05 / 12.42 W |

质量指标：

```text
JSON validity       1.00
risk accuracy       0.35
event precision     0.00
event recall        0.00
event micro-F1      0.00
unsafe advice rate  0.00
```

INT4 输出倾向于统一预测 `vehicle_near_maneuver_path`，导致 16 个假阳性，并漏掉全部
`narrow_passage` 与 `visibility_occlusion` 正样本。流程部署成功不等于量化质量可接受。

### 24.15 Jetson FP16 与 INT4 AWQ 同机对比

两组实验使用相同 Jetson、模型 revision、Edge-LLM revision、冻结 workload、20 样本测试
集、输入上限 768、KV 1024 和 15 W power mode。主要对比：

| 指标 | Edge-LLM FP16 | Edge-LLM INT4 AWQ | 变化 |
| --- | ---: | ---: | ---: |
| engine 大小 | 3453786212 B | 1362769140 B | -60.5% |
| mean 延迟 | 53638.78 ms | 10685.66 ms | 5.02x 加速 |
| p50 延迟 | 50753.15 ms | 10524.38 ms | 4.82x 加速 |
| p90 延迟 | 69082.40 ms | 11394.11 ms | 6.06x 加速 |
| p99 延迟 | 75081.88 ms | 12626.09 ms | 5.95x 加速 |
| aggregate tokens/s | 1.4803 | 7.3276 | 4.95x |
| RAM peak | 7418 MiB | 5072 MiB | -31.6% |
| swap peak | 1904 MiB | 840 MiB | -55.9% |
| GPU 温度 peak | 65.03 C | 60.75 C | -4.28 C |
| JSON validity | 1.00 | 1.00 | 持平 |
| risk accuracy | 0.35 | 0.35 | 持平 |
| event micro-F1 | 0.3590 | 0.0000 | 明显退化 |
| unsafe advice rate | 0.00 | 0.00 | 持平 |

当前 INT4 路径完成了 `checkpoint -> AWQ -> ONNX -> TensorRT engine -> Jetson runtime ->
StudyReport` 的完整闭环，并在延迟、吞吐、engine 大小和内存上获得明确收益。但 128 条
通用新闻文本只覆盖语言统计，不覆盖泊车视觉指令分布；本实验的事件 F1 退化说明该校准
方案只能作为部署与性能验证，不能作为最终质量版本。下一轮质量优化应使用冻结测试集之外
的泊车场景图文校准集，并保持相同测试集复测，不能用测试样本参与校准。

### 24.16 本轮证据与收尾状态

本机忽略目录保存了量化、导出、构建、串行冒烟、正式 StudyReport、服务器日志和
`tegrastats` 原始证据：

```text
reports/autodl-int4-20260813/
reports/jetson-runtime-20260813-int4/
```

远端临时 HTTP Range 服务、SSH 隧道、Edge-LLM 服务和 `tegrastats` 进程均已停止。
AutoDL 数据盘仍保留固定模型、量化 checkpoint、ONNX 和 venv，实例可在确认无需继续实验
后关机以停止 GPU 计费。

## 25. 领域 LoRA/SFT 实测闭环（2026-08-13）

### 25.1 目标与数据边界

在获得本轮 GPU 服务器训练授权后，LoRA 从示例占位符推进为真实可执行流程。冻结的
`ps20_pilot_v1` 20 张图片继续只作为 test，未参与训练、验证、阈值选择或 adapter 合并。

本机 PS2.0 原始数据盘点：

```text
training JPG    9827
testing JPG     4676
pilot JPG         20
```

从 `training` 文件名提取连续采集来源组，例如 `p2_img28_0408` 归入 `p2_img28`。固定
随机种子 42 后，每个来源组只选一张图，得到：

```text
train          64 images / 64 source groups
validation     16 images / 16 source groups
test           20 pilot images / independent pilot groups
train-val overlap   0
train-test overlap  0
```

PS2.0 `.mat` 是泊车位几何标注，不等同于本项目的风险 JSON，因此没有将其伪装成人工
风险标签。使用固定基础模型、固定 workload 对 80 张图片生成弱监督 JSON；80/80 通过
`ParkingAssessment.from_mapping` 严格校验，失败文件为空。

弱监督标签分布：

```text
risk_level low               80
narrow_passage               65
vehicle_near_maneuver_path   19
maintain_observation         80
slow_down                     4
```

该分布说明 teacher 本身存在 low 与 `narrow_passage` 偏置；因此本轮能够证明训练、合并、
复测和导出链路，但不能将弱监督结果描述为人工精标或真实道路安全能力。

数据生成命令：

```bash
cd /root/autodl-tmp/JetsonVLM
export PYTHONPATH=/root/autodl-tmp/JetsonVLM/src

.venv-lora/bin/python scripts/generate_lora_dataset.py \
  --image-root data/processed/lora/source_images \
  --workload configs/workloads/parking_risk_v1.json \
  --model models/Qwen3-VL-2B-Instruct-89644892 \
  --model-revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --output data/processed/lora/ps80_teacher_v1.jsonl \
  --train-count 64 --validation-count 16 --seed 42
```

### 25.2 独立 LoRA 训练环境

INT4 环境中的 PyTorch `2.12.0+cu130` 未包含 RTX 4090 D `sm_89` 的可执行 kernel，
项目原架构检查在加载模型前明确拒绝。服务器 base 环境的 PyTorch `2.8.0+cu128` 虽然
`get_arch_list()` 未逐项列出 `sm_89`，但实际 CUDA tensor kernel 成功。架构检查因此
修正为：精确架构缺失时执行真实 CUDA kernel probe，以可执行事实作为最终判据。

没有修改已验证的 INT4 venv，而是建立独立环境：

```bash
/root/miniconda3/bin/python -m venv --system-site-packages \
  /root/autodl-tmp/JetsonVLM/.venv-lora

.venv-lora/bin/python -m pip install \
  transformers==5.9.0 peft==0.18.0 accelerate==1.10.1
```

最终环境：

```text
GPU           RTX 4090 D 24 GiB, capability 8.9
PyTorch       2.8.0+cu128
Transformers  5.9.0
PEFT          0.18.0
Accelerate    1.10.1
CUDA kernel   passed
```

### 25.3 LoRA 配置与训练实测

只对语言模型 28 层 self-attention 的 `q_proj/k_proj/v_proj/o_proj` 注入 LoRA；视觉塔、
LM head 和基础权重冻结：

```text
rank                    16
alpha                   32
dropout                 0.05
learning rate           1e-4
precision               BF16
epochs                  1
gradient accumulation   4
optimizer steps         16
```

可审计执行命令：

```bash
cd /root/autodl-tmp/JetsonVLM
export PATH=/root/autodl-tmp/JetsonVLM/.venv-lora/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/JetsonVLM/src

.venv-lora/bin/python scripts/train_lora.py \
  --config configs/flows/train_qwen3_vl_2b_lora_ps80_v1.json \
  --execute
```

flow 状态为 `succeeded`，实际训练结果：

```text
train samples             64
validation samples        16
trainable parameters      6422528
total parameters          2133954560
trainable ratio           0.00300968 (0.301%)
validation loss           0.0845192
peak CUDA memory          5.25345 GiB
elapsed                   21.5421 s
```

adapter：

```text
directory  artifacts/adapters/qwen3_vl_2b_parking_lora_ps80_v1
size       about 36 MiB
weight sha256
3155886d4db78683ab69a6eea342d52c606f14f2e3214157590ed3809af713cf
```

本机另保存压缩归档：

```text
reports/autodl-lora-20260813/qwen3_vl_2b_parking_lora_ps80_v1_adapter.tar.gz
bytes   25839843
sha256  1079971ffd8227801f640c0b88f865b1eeb9e0e7179c0035a21b480b7f56f20a
```

### 25.4 Base、LoRA 与合并模型质量对比

服务器使用同一 RTX 4090 D、BF16、模型 revision、prompt、输入尺寸、生成参数和
`ps20_pilot_v1` test 集分别运行独立进程：

```bash
PYTHONPATH=src .venv-lora/bin/python -m parksight_vlm.app.run_study \
  --config configs/studies/server_transformers_base_ps20_pilot.json

PYTHONPATH=src .venv-lora/bin/python -m parksight_vlm.app.run_study \
  --config configs/studies/server_transformers_lora_ps20_pilot.json

PYTHONPATH=src .venv-lora/bin/python -m parksight_vlm.app.run_study \
  --config configs/studies/server_transformers_merged_lora_ps20_pilot.json
```

结果：

| 指标 | Base | LoRA adapter | merged LoRA |
| --- | ---: | ---: | ---: |
| backend completed | 20/20 | 20/20 | 20/20 |
| strict JSON | 100% | 100% | 100% |
| risk accuracy | 0.35 | 0.35 | 0.35 |
| event precision | 0.2917 | 0.3500 | 0.3500 |
| event recall | 0.4375 | 0.4375 | 0.4375 |
| event micro-F1 | 0.3500 | 0.3889 | 0.3889 |
| unsafe advice rate | 0 | 0 | 0 |

LoRA 的事件 micro-F1 绝对提升 `0.0389`、相对提升约 `11.1%`；
`vehicle_near_maneuver_path` 假阳性从 6 降为 0。与此同时，`narrow_passage` 假阳性
从 11 增到 13，`visibility_occlusion` 仍有 9 个假阴性。该结果是小样本弱监督 LoRA
对一类误差有效、对其他类别仍不足的真实证据，不描述为全面质量提升。

### 25.5 adapter 合并与 ONNX 导出

合并命令：

```bash
.venv-lora/bin/python scripts/merge_lora.py \
  --config configs/flows/merge_qwen3_vl_2b_lora_ps80_v1.json \
  --execute
```

结果：

```text
flow status         succeeded
flow elapsed        24.25 s
merged checkpoint   about 4.0 GiB
merged weight sha256
5d4b5c3f615b5ebc20dde610029fd5f3ff747338796c85a302b34ce5b0e13e4c
```

合并模型的 test 质量指标与未合并 adapter 完全一致；部分 `evidence` 文本措辞不同，
但风险、事件、JSON 和安全建议聚合指标一致。

TensorRT Edge-LLM 导出：

```bash
export PATH=/root/autodl-tmp/TensorRT-Edge-LLM/.venv-int4/bin:$PATH

python scripts/export_model.py \
  --config configs/flows/export_qwen3_vl_2b_lora_ps80_v1.json \
  --execute
```

flow 状态为 `succeeded`，用时约 112 秒。LLM 主要文件：

```text
model.onnx sha256
e736aff8fa1216dabb9d9a864c608fdad4881858ff55229f1ad12579a8012807

model.onnx.data sha256
9db77500e191811f6342589b820a5db6b307d6e841a71b5536ef7cbff04eaf11

embedding.safetensors sha256
af9dbce9b96ca45a1cdc7c5ebdd18dc780567e707388e7cead78fbf1adb256d5
```

只归档发生变化的 LLM ONNX，visual engine 可复用相同基础 revision 的既有 FP16 产物：

```text
archive bytes   4078346240
archive sha256  5053f18b8a86251d070a2d8267f0fcd8adf578d2f0c09e614a0fcd4cdc078f96
```

### 25.6 LoRA ONNX 传输与 Jetson engine 构建

板端恢复连接后，通过服务器回环 HTTP Range 服务、四路 SSH 本地转发和 Jetson 侧
`aria2c` 断点续传 4.078 GB LLM ONNX 归档。16 个 Range 连接实际分布到 4 条 SSH
会话；公网链路平均速度为 2.9 MiB/s。完成条件不是稀疏文件显示的逻辑大小，而是
`.aria2` 状态文件消失、字节数和 SHA-256 均匹配：

```text
archive bytes   4078346240
archive sha256  5053f18b8a86251d070a2d8267f0fcd8adf578d2f0c09e614a0fcd4cdc078f96
download status OK
```

解包后 `config.json`、`embedding.safetensors`、`model.onnx`、`model.onnx.data`、chat
template 和 tokenizer 文件逐项通过 `llm.sha256`。板端构建命令对应配置：

```bash
.venv-jetson/bin/python scripts/build_engine.py \
  --config configs/flows/build_qwen3_vl_2b_lora_ps80_v1_llm_engine_i768_k1024.json \
  --execute
```

构建口径为 batch 1、input 768、KV cache 1024、FP16、weight streaming；只重建发生变化
的 LLM engine，复用已验证的 visual engine。构建实测：

```text
flow status       succeeded
declared outputs  6/6
started           2026-08-13T12:10:40Z
finished          2026-08-13T12:14:37Z
llm.engine bytes  3453786212
llm.engine sha256 d38adc5d532615d7183a6b4aa8413020bd76a5991e49ecd51ca84d0442334224
visual sha256     3c6b4cce682e021b09c066d0e325335e31ef9edbf613c754be586035c26f5c2f
```

builder 峰值 RAM 接近 7.45/7.62 GB，swap 最高观察到约 8.2/12.0 GB，临时 8 GiB 磁盘
swap 避免了构建阶段 OOM。构建结束后保留 ONNX、Base/LoRA/INT4 engine，没有删除既有
实验产物。

### 25.7 runtime OOM 定位与 headless 运行条件

构建后首次加载 LoRA runtime 时，LLM engine、weight streaming budget 0 和 1.348 GB
execution context 均成功，但视觉 engine 申请 811032832 字节时出现：

```text
NvMapMemAllocInternalTagged: 1075072515 error 12
Requested amount of GPU memory (811032832 bytes) could not be allocated
No valid multimodal engine found
```

依次完成以下判别：

1. 清 page cache 后 LoRA 仍失败，排除普通文件缓存；
2. 清空 zram 后 LoRA 仍失败，排除构建遗留 swap 页；
3. 停止非推理服务后 LoRA 仍失败，排除后台进程 RSS 是唯一原因；
4. 同状态加载已知成功的 Base FP16 engine 也在同一 811 MB 分配处失败，排除 LoRA
   engine 特有内存回归；
5. 重启后图形桌面状态仍失败，说明 `free -h` 的 available 不能代表 NvMap 所需的连续
   统一内存块。

随后临时执行：

```bash
sudo systemctl isolate multi-user.target
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory'
```

显示栈退出后 NvMap clients 从 Xorg/GNOME/portal 共约 34 MB 降为 0，`tegrastats` 空闲
RAM 降到约 509/7619 MB。此时 LoRA 双 engine 成功加载，HTTP `/health` 返回
`healthy`，视觉 runner 初始化和 decode CUDA graph 捕获均成功。评测结束后已关闭
server，并执行 `sudo systemctl isolate graphical.target`；图形桌面、Docker、Snap、
fwupd 和 PackageKit 均恢复为 `active`。8 GiB 临时磁盘 swap 继续启用。

### 25.8 Jetson LoRA 单图与冻结 20 样本实测

同一图片连续三次 smoke 均为严格 JSON 成功，输出一致；端到端分别为：

```text
run1  32581.889 ms  47 tokens
run2  32504.516 ms  47 tokens
run3  32509.464 ms  47 tokens
```

随后沿用与 Base FP16 相同的模型 commit、workload identity、`ps20_pilot_v1` test 集、
input 768、KV 1024 和 power mode 运行 20 样本 Study：

| 指标 | Jetson LoRA 实测 |
| --- | ---: |
| 后端完成 / 严格 JSON | 20/20 / 20/20 |
| 失败汇总 | 空 |
| 风险等级准确率 | 35% |
| 事件 micro precision / recall / F1 | 0.3500 / 0.4375 / 0.3889 |
| 不安全建议率 | 0% |
| 端到端 mean / p50 / p90 / p99 | 32.01 / 30.60 / 33.08 / 40.90 s |
| 输出 token 总数 / 聚合速率 | 924 / 1.443 token/s |
| RAM / swap 峰值 | 7351 / 580 MB |
| GPU 利用率均值 | 98.20% |
| 输入功耗均值 | 10.10 W |
| GPU 峰温 | 61.94°C |

同板 Base FP16 的事件 micro-F1 为 0.3590，LoRA 提升到 0.3889；车辆事件假阳性由 3
降为 1，狭窄通道假阳性由 13 降为 12，但 9 个遮挡事件仍全部漏检。平均端到端延迟
相对 Base 从 53.64 秒降至 32.01 秒，但 LoRA 输出 token 总数也从 1588 降至 924
（减少 41.8%），且聚合输出速率从 1.480 略降至 1.443 token/s。因此该延迟下降主要
来自更短输出，不能表述为 LoRA engine 带来的运行时加速。

原始证据已下载到本机忽略目录：

```text
reports/jetson-lora-20260813/
  build_qwen3_vl_2b_lora_ps80_v1_llm_engine_i768_k1024.{json,log}
  jetson_edgellm_lora_ps80_v1_ps20_pilot_i768_k1024.json
  jetson_edgellm_lora_ps80_v1_ps20_pilot_i768_k1024_summary.json
  lora_smoke_serial_run{1,2,3}_20260813.json
  edgellm_server_lora_ps80_v1_headless_20260813.log
  tegrastats_lora_engine_build_20260813.log
  tegrastats_lora_ps80_v1_20sample_20260813.log
```

至此 LoRA 已完成数据划分、弱监督样本、真实训练、合并、服务器质量对照、Edge-LLM
ONNX 导出、Jetson engine 构建、双 engine runtime、单图稳定性和冻结 20 样本评测闭环。

## 26. 当前最终状态快照（文档更新于 2026-08-16）

本节汇总当前仓库与最近一次外部运行证据。文档整理日期为 2026-08-16；最近一次 GPU
服务器和 Jetson 实际运行验证发生在 2026-08-13，未把文档更新时间表述为新的硬件
复测日期。

### 26.1 仓库与验证状态

当前主分支最近一次项目提交：

```text
branch  main
commit  3e55b9a docs: record jetson lora deployment results
remote  origin/main 已包含该提交
```

提交前验证结果：

```text
PYTHONPATH=src .venv/Scripts/python.exe -m unittest discover -s tests
Ran 44 tests in 0.137s
OK

git diff --check
passed
```

本机 `uv` 曾因全局缓存目录 ACL 无法初始化，因此使用仓库 `.venv` 与显式
`PYTHONPATH=src` 完成等价测试；这不是项目测试失败，也没有为验证修改依赖环境。
`docs/record.md` 与 `reports/` 按 `.gitignore` 保留为本机过程记录和原始证据，不进入
Git 提交。

### 26.2 已完成的端到端链路

当前项目已形成并实际执行以下业务链路：

```text
ParkingCase
  -> RiskRuntime
  -> TensorRT Edge-LLM HTTP
  -> ParkingAssessment 严格 JSON
  -> InferenceRecord
  -> StudyReport / Jetson runtime summary
```

已完成的外部实验阶段：

1. 固定 `Qwen/Qwen3-VL-2B-Instruct` 不可变 revision；
2. 服务器 Transformers Base 正确性与质量参考；
3. Jetson Transformers FP16 可运行性检查，并保留 OOM 失败事实；
4. TensorRT Edge-LLM Base FP16 ONNX、LLM/visual engine、HTTP runtime 和 20 样本 Study；
5. 80 张独立泊车图片的弱监督数据构造与 64/16 来源组隔离；
6. RTX 4090 D LoRA 训练、adapter 质量评测、合并和合并模型复测；
7. LoRA 合并模型 Edge-LLM ONNX 导出、Jetson engine 构建、3/3 smoke 和 20 样本 Study；
8. LLM backbone INT4 AWQ 量化、ONNX、Jetson engine 和 20 样本 Study；
9. Base FP16、LoRA 与 INT4 的质量、延迟、输出速率、内存、功耗和温度证据汇总。

因此，项目不再处于“仅有格式/schema/脚本、尚未涉及模型”的阶段，也不再是“只完成
ONNX 导出但没有板端 runtime”的阶段。Base FP16、LoRA 合并模型和 INT4 均有真实
Jetson engine 与请求路径证据。

### 26.3 三组板端实验的最终对比

三组实验均使用固定 20 张 `ps20_pilot_v1` test 集；服务器数据不替代板端性能结论。

| 指标 | Base FP16 | LoRA FP16 merged | INT4 AWQ |
| --- | ---: | ---: | ---: |
| 后端完成 | 20/20 | 20/20 | 20/20 |
| 严格 JSON 有效率 | 100% | 100% | 100% |
| 风险等级准确率 | 35% | 35% | 35% |
| 事件 micro-F1 | 0.3590 | 0.3889 | 0 |
| 平均端到端时延 | 53.64 s | 32.01 s | 10.69 s |
| p50/p90/p99 | 50.75/69.08/75.08 s | 30.60/33.08/40.90 s | 见 INT4 summary |
| 聚合输出速率 | 1.480 token/s | 1.443 token/s | 7.33 token/s |
| RAM 峰值 | 7418 MB | 7351 MB | 约降低 31.6% |
| swap 峰值 | 1904 MB | 580 MB | 见 INT4 summary |
| GPU 利用率均值 | 97.39% | 98.20% | 见 INT4 summary |
| 输入功耗均值 | 10.05 W | 10.10 W | 见 INT4 summary |
| GPU 峰温 | 65.03°C | 61.94°C | 见 INT4 summary |

解释边界：

- LoRA 的事件 micro-F1 相对板端 Base FP16 从 0.3590 提升到 0.3889，但风险准确率仍为
  35%，9 个 `visibility_occlusion` 仍全部漏检；不能描述为业务质量已经达标。
- LoRA 平均延迟下降 40.3%，但输出 token 总数也减少 41.8%，聚合 token/s 从 1.480
  略降至 1.443，因此不能把更短端到端时间描述成 LoRA runtime 加速。
- INT4 的 engine 体积、时延、输出速率和 RAM 指标证明量化部署收益，但事件 micro-F1
  退化为 0；当前通用文本校准版本是成功的部署实验和失败的质量版本，不是推荐模型。
- Jetson Transformers FP16 仍没有同一 20 样本成功报告，因此不能给出 Edge-LLM 相对
  Transformers 的严格同机加速比；早期 OOM 是需要保留的实验结论。

### 26.4 8 GB Jetson 的可复现运行条件

Base FP16/LoRA 双 engine 在图形桌面状态下可能出现：

```text
Requested amount of GPU memory (811032832 bytes) could not be allocated
NvMapMemAllocInternalTagged ... error 12
```

该问题已通过 Base engine 对照确认是统一内存连续块/NvMap 状态问题，不是 LoRA engine
特有回归。最近一次成功运行使用：

```bash
sudo systemctl isolate multi-user.target
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory'
```

随后以 `--weight-streaming-budget-bytes 0` 启动 Edge-LLM。评测结束后执行：

```bash
sudo systemctl isolate graphical.target
```

收尾检查中 graphical target、display manager、Docker、Snap、fwupd 和 PackageKit 均为
`active`，推理 server、Study、`tegrastats`、`aria2c`、临时代理和 SSH 传输隧道均已
停止。3.7 GiB zram 与 8 GiB `/home/ubuntu/parksight-build.swap` 当时均处于启用状态；
磁盘 swap 未写入 `fstab`，重启后需要显式重新启用。

### 26.5 证据位置

```text
reports/jetson-runtime-20260812/       Base FP16 最终 Study 与遥测
reports/jetson-runtime-20260813-int4/  INT4 engine、Study 与遥测
reports/autodl-lora-20260813/          LoRA 训练、服务器评测与 adapter 归档
reports/jetson-lora-20260813/          LoRA Jetson 构建、server、smoke、Study 与遥测
```

可提交的项目状态与说明位于：

```text
README.md
docs/status.md
docs/progress.md
docs/execution-report.md
docs/resume-project.tex
```

### 26.6 当前项目边界与后续可选工作

以“理解端侧多模态推理、LoRA、量化、TensorRT engine 和完整评测链路”为目标，当前
项目主链路已经闭环，可以用于简历和面试讲解。尚未完成或不应夸大的内容：

1. 弱监督 80 样本不能代表真实泊车安全数据规模，后续应增加独立人工精标；
2. INT4 需要使用冻结测试集之外的泊车图文校准集重新量化；
3. Jetson Transformers FP16 的同口径成功基线仍缺失；
4. 当前是单图离线风险分析，不包含摄像头实时视频、车辆控制或闭环 VLA；
5. 尚未以 profiling 证据实现独立 CUDA 预处理优化，不能声称自研 CUDA kernel 已带来
   端到端加速。

如果继续迭代，优先级建议为：人工精标与校准数据质量 > Transformers 基线或明确 OOM
验收 > 分阶段 profiling > 有证据的 CUDA 预处理/内存优化。若以当前版本投递简历，
应采用 `docs/resume-project.tex` 中的客观描述，并能解释 INT4 质量退化、LoRA 短输出
与运行时加速的区别，以及 headless/compaction 解决的 NvMap 连续内存问题。

## 27. 2026-08-17：候选标注、独立校准集与 Transformers 分阶段 profiling

### 27.1 本轮目标与 GPU 服务器需求

本轮只处理两项工作：

1. 整理 PS2.0 的 80 条开发样本，修正 teacher 标签偏置并划出独立 INT4 校准集；
2. 在 Jetson 补齐 Transformers FP16 的阶段 profiling 和当前可运行边界。

这两项不需要启动 4090 服务器。4090 只在使用新标签重新训练 LoRA、使用新校准集重新
量化、重新导出 ONNX 时需要。

### 27.2 弱监督标签盘点

原始 teacher 文件为：

```text
reports/autodl-lora-20260813/ps80_teacher_v1.jsonl
```

盘点得到 80 个唯一 `source_group_id`，原划分为 64 train / 16 validation。标签分布
存在明显偏置：80/80 的 `risk_level` 都是 `low`，65 条包含 `narrow_passage`，只有
19 条包含 `vehicle_near_maneuver_path`。

使用以下命令生成五张 4x4 联系表：

```powershell
& '<BUNDLED_PYTHON>' scripts\build_label_review_sheets.py `
  --records data\manifests\ps80_development_v1.jsonl `
  --image-root data\processed\lora\source_images `
  --output-directory reports\label-review-20260817
```

本轮逐图形成 `data/annotations/ps80_reviewed_v1.jsonl`。标签来源明确记为
`codex_visual_review_v1_single_pass`：它用于修正明显 teacher 偏置和建立下一轮实验
入口，不是人工双人金标，也不能描述为真实道路安全能力已经得到人工验收。联系表保留
用于后续人工终审。

相对 teacher 的字段变化：

| 项目 | 数量 |
| --- | ---: |
| 风险等级变化 | 33/80 |
| 事件集合变化 | 77/80 |
| 任意 assessment 字段变化 | 80/80 |

### 27.3 无泄漏拆分与校准记录

校准选择固定在：

```text
configs/data/ps16_int4_calibration_v1.json
```

它从原 64 条 train 候选中按低风险、车辆近场、固定障碍和可见性类别选择 16 个来源
组。生成命令：

```powershell
$env:PYTHONPATH='src'
& '.\.venv\Scripts\python.exe' scripts\prepare_reviewed_lora_dataset.py `
  --development-manifest data\manifests\ps80_development_v1.jsonl `
  --weak-annotations data\annotations\ps80_teacher_v1.jsonl `
  --annotations data\annotations\ps80_reviewed_v1.jsonl `
  --calibration-config configs\data\ps16_int4_calibration_v1.json `
  --image-root data\processed\lora\source_images `
  --workload configs\workloads\parking_risk_v1.json `
  --frozen-test-manifest data\manifests\ps20_pilot_v1.jsonl `
  --lora-output data\processed\lora\ps64_reviewed_v1.jsonl `
  --calibration-output data\processed\calibration\ps16_int4_calibration_v1.jsonl `
  --summary-output reports\data\ps80_reviewed_v1_summary.json
```

结果：

| 划分 | 样本数 |
| --- | ---: |
| LoRA train | 48 |
| LoRA validation | 16 |
| INT4 calibration | 16 |
| 合计 | 80 |

校验结果：LoRA 与 calibration 交集为 0，开发来源组与冻结 20 样本 test 的交集为 0。
未来 LoRA 入口为
`configs/training/qwen3_vl_2b_lora_ps64_reviewed_v1.json`。

INT4 当前只量化 LLM backbone，视觉 engine 保持 FP16。因此 calibration JSONL 的
`text` 字段使用固定泊车 system prompt、完整用户约束和结构化答案覆盖领域 token
分布；图片路径和 assessment 同时保留用于追溯。这是泊车领域语言骨干校准，不应称为
视觉编码器量化校准。现有 INT4 engine 仍来自旧的 128 条新闻文本，尚未用新集重新
量化。

### 27.4 profiling 代码与验证

`HuggingFaceQwen3VlBackend` 新增可选 `profile_stages`。开启时在上游模型的
`visual` 和 `language_model` 模块安装 forward hook：

- `visual` forward 总时延记为 `vision_encode_ms`；
- 语言模型第一次 forward 记为 `prefill_ms`；
- 后续语言模型 forward 之和记为 `decode_ms`；
- 找不到模块时字段保持 `null`，不推测数据。

hook 边界调用 `torch.cuda.synchronize()`，因此该模式会改变绝对时延，只能作为阶段
占比诊断。独立配置为：

```text
configs/studies/jetson_transformers_fp16_profile_ps20_pilot.json
```

本机验证：

```powershell
$env:PYTHONPATH='src'
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests
git diff --check
```

结果为 46 个测试全部通过，`git diff --check` 通过。

### 27.5 Jetson 执行命令

先确认当前状态：graphical target 和 display manager 均为 active，8 GiB 文件 swap 与
约 3.7 GiB zram 均启用，固定模型 commit 缓存存在。Python 导入必须带 cuDSS/CUDA
动态库路径：

```bash
cd /home/ubuntu/JetsonVLM
LD_LIBRARY_PATH=/home/ubuntu/JetsonVLM/.venv-jetson/lib/python3.10/site-packages/nvidia/cu12/lib:/usr/local/cuda/lib64 \
  .venv-jetson/bin/python -c 'import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())'
```

实测为 `torch 2.9.1`、`transformers 4.57.6`、CUDA 可用。按板端不更新 Git、只同步
修改文件的约定，SCP 同步 `transformers.py`、`runtime_factory.py` 和 profiling study
配置。运行命令：

```bash
cd /home/ubuntu/JetsonVLM
mkdir -p reports/jetson-transformers-profile-20260817
tegrastats --interval 1000 \
  > reports/jetson-transformers-profile-20260817/tegrastats.log 2>&1 &
TEGRSTATS_PID=$!

LD_LIBRARY_PATH=/home/ubuntu/JetsonVLM/.venv-jetson/lib/python3.10/site-packages/nvidia/cu12/lib:/usr/local/cuda/lib64 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src \
timeout 1200s \
.venv-jetson/bin/python -m parksight_vlm.app.run_study \
  --config configs/studies/jetson_transformers_fp16_profile_ps20_pilot.json \
  > reports/jetson-transformers-profile-20260817/stdout.log \
  2> reports/jetson-transformers-profile-20260817/stderr.log

kill "$TEGRSTATS_PID"
```

退出码为 0。运行完成后单独加载相同模型并打印 `hf_device_map`，结果为：

```text
Counter({'0': 1})
{'': 0}
```

因此最终模型完整映射到 `cuda:0`。加载过程出现的 meta/CPU offload 警告不能单独作为
最终 offload 结论。

### 27.6 20 样本 profiling 结果

后端完成和严格 JSON 均为 20/20，失败汇总为空。阶段时延：

| 阶段 | p50 | p90 | p99 |
| --- | ---: | ---: | ---: |
| preprocess | 27.60 ms | 33.86 ms | 160.35 ms |
| vision encode | 225.11 ms | 232.16 ms | 747.24 ms |
| prefill | 626.51 ms | 633.86 ms | 802.51 ms |
| decode | 17.04 s | 26.64 s | 27.56 s |
| instrumented generate | 19.24 s | 29.82 s | 30.75 s |
| instrumented end-to-end | 19.27 s | 29.86 s | 45.12 s |

decode p50 约占插桩 generate p50 的 88.5%，单输出 token 的 decode 中位耗时约
226.3 ms。预处理 p50 仅 27.6 ms，当前主要瓶颈是 LLM decode，不是 Pillow resize 或
Processor 输入构造。未插桩基线仍使用既有报告中的端到端 p50 9.38 s；不能拿插桩后的
19.27 s 计算运行时加速比。

451 条 tegrastats 的摘要：

| 指标 | 结果 |
| --- | ---: |
| RAM 峰值 | 7302 / 7619 MB |
| swap 峰值 | 1174 / 12002 MB |
| 最小 lfb | 1 x 2 MB |
| GPU 利用率均值 / 峰值 | 58.55% / 99% |
| 输入功耗均值 / 峰值 | 8.77 / 12.12 W |
| GPU 温度均值 / 峰值 | 57.91 / 60.97 C |

当前 R36.5.0、graphical target、swap 已启用的配置在极小连续内存余量下仍完成了完整
FP16 study，属于明确的“可运行但余量很小”边界。早期 R36.4.7/旧环境中模型加载阶段
的 NvMap error 12、GR3D 0% 和 OOM 记录仍作为失败边界保留。swap 可以缓解可换页
内存压力，但不能保证 NvMap 所需连续物理块，因此不能仅凭 `free -h` 或 swap 开关
解释 OOM。

本轮原始证据保存在忽略目录：

```text
reports/label-review-20260817/
reports/data/ps80_reviewed_v1_summary.json
reports/jetson-transformers-profile-20260817/study.json
reports/jetson-transformers-profile-20260817/tegrastats.log
reports/jetson-transformers-profile-20260817/summary.json
reports/jetson-transformers-profile-20260817/stdout.log
reports/jetson-transformers-profile-20260817/stderr.log
```

## 28. 2026-08-17：复核数据 LoRA 重训与领域 INT4 端到端复测

### 28.1 本轮范围与结论

本轮在一台 RTX 4090 D 服务器上完成两项计划工作，并在同一 Jetson Orin Nano 上
完成最终部署复测：

1. 使用 `ps64_reviewed_v1` 重新训练、评测并合并 LoRA；
2. 使用独立的 `ps16_int4_calibration_v1` 重新执行 LLM backbone INT4 AWQ、ONNX
   导出、Jetson engine 构建和冻结 20 样本 study。

两条流程均从配置入口运行到实际产物与报告，部署链路完整。但质量结果是混合/负向：

- 经过非 low 样本过采样的 LoRA adapter 达到 20/20 严格 JSON、风险准确率 50%、事件
  micro-F1 0.182；风险准确率高于旧 LoRA，但事件 F1 低于旧弱监督 LoRA 的 0.389；
- 合并 checkpoint 的同口径结果为风险准确率 45%、事件 micro-F1 0.100，和 adapter
  在线加载结果并不完全相同；
- 新领域校准 INT4 在 Jetson 上 20/20 后端完成，但严格 JSON 仅 4/20。16 条失败输出
  基本都是被 Markdown `json` 代码围栏包裹的完整 JSON，违反了严格输出协议；
- 新 INT4 的速度与旧通用文本校准 INT4 基本相同，没有形成新的性能收益，且格式遵循
  明显退化。因此它是“量化部署成功、质量验收失败”的负向实验，不替换当前已保留的
  旧 INT4 对照结果。

复核标注来源仍为 `codex_visual_review_v1_single_pass`，不是人工双人金标。本轮结果只能
用于验证训练、量化和部署方法及发现数据问题，不能解释为泊车安全能力已经完成验收。

### 28.2 新 GPU 服务器环境

首先进行只读环境检查：

```bash
. /etc/os-release
printf '%s\n' "$PRETTY_NAME"
uname -m
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
  --format=csv,noheader
python3 -V
git --version
df -h /root/autodl-tmp
free -h
python3 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

实测环境为 Ubuntu 22.04.5 x86_64、RTX 4090 D 24564 MiB、驱动 595.71.05、系统
Python 3.12.3、PyTorch 2.8.0+cu128，CUDA 可用。项目位于：

```text
/root/autodl-tmp/JetsonVLM
```

训练环境与量化工具环境相互隔离：

```bash
cd /root/autodl-tmp/JetsonVLM
python3 -m venv --system-site-packages .venv-train
PIP_CACHE_DIR=/root/autodl-tmp/pip-cache \
  .venv-train/bin/python -m pip install --upgrade pip
PIP_CACHE_DIR=/root/autodl-tmp/pip-cache \
  .venv-train/bin/python -m pip install \
  'transformers==5.9.0' 'peft==0.18.0' 'accelerate==1.10.1' \
  'safetensors>=0.6.2'

cd /root/autodl-tmp
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git checkout 7f061f21f0a581ba234a1e233c9315b89d8e47d6
/root/miniconda3/bin/python -m venv .venv-int4
.venv-int4/bin/python -m pip install -e '.[tools]'
.venv-int4/bin/python -m pip check
```

最终版本：

| 环境 | 关键包 |
| --- | --- |
| `.venv-train` | torch 2.8.0+cu128、transformers 5.9.0、peft 0.18.0、accelerate 1.10.1 |
| `.venv-int4` | torch 2.12.0+cu130、torchvision 0.27.0、transformers 5.9.0、modelopt 0.44.0、datasets 4.8.5 |

量化工具仓库严格固定在
`7f061f21f0a581ba234a1e233c9315b89d8e47d6`，`pip check` 通过。

### 28.3 固定模型下载与校验

基础模型仍使用同一不可变 revision：

```text
Qwen/Qwen3-VL-2B-Instruct
89644892e4d85e24eaac8bacfd4f463576704203
```

标准下载入口为：

```bash
cd /root/autodl-tmp/JetsonVLM
mkdir -p models /root/autodl-tmp/hf-home
HF_HOME=/root/autodl-tmp/hf-home \
  .venv-train/bin/hf download Qwen/Qwen3-VL-2B-Instruct \
  --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --local-dir models/Qwen3-VL-2B-Instruct-89644892
```

本次 Hugging Face 大权重连接曾停滞，因此安装 `aria2` 并对同一固定 revision 的权重
URL 断点续传。最终校验结果：

| 文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `model.safetensors` | 4255140312 | `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0` |

下载恢复手段不改变模型身份；后续训练、量化和导出均读取该固定目录。

### 28.4 LoRA 训练入口与类别采样修正

新增/使用的配置入口：

```text
configs/training/qwen3_vl_2b_lora_ps64_reviewed_v1.json
configs/flows/train_qwen3_vl_2b_lora_ps64_reviewed_v1.json
configs/flows/merge_qwen3_vl_2b_lora_ps64_reviewed_v1.json
configs/studies/server_transformers_lora_ps64_reviewed_v1_ps20_pilot.json
configs/studies/server_transformers_merged_lora_ps64_reviewed_v1_ps20_pilot.json
```

训练脚本增加确定性的 `non_low_oversampling_factor`。它只复制 train split 中风险等级
不是 `low` 的记录，不改变 validation，也不会把 calibration/test 数据引入训练。配置
使用 factor 2，将 48 个唯一训练样本扩展为 63 个有效训练记录。

服务器执行命令：

```bash
cd /root/autodl-tmp/JetsonVLM
export PATH=/root/autodl-tmp/JetsonVLM/.venv-train/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/JetsonVLM/src

python scripts/train_lora.py \
  --config configs/flows/train_qwen3_vl_2b_lora_ps64_reviewed_v1.json
python scripts/train_lora.py \
  --config configs/flows/train_qwen3_vl_2b_lora_ps64_reviewed_v1.json \
  --execute

python -m parksight_vlm.app.run_study \
  --config configs/studies/server_transformers_lora_ps64_reviewed_v1_ps20_pilot.json

python scripts/merge_lora.py \
  --config configs/flows/merge_qwen3_vl_2b_lora_ps64_reviewed_v1.json \
  --execute
python -m parksight_vlm.app.run_study \
  --config configs/studies/server_transformers_merged_lora_ps64_reviewed_v1_ps20_pilot.json
```

为避免只观察训练损失，本轮保留三轮对照：

| 训练版本 | 唯一/有效 train | epoch / step | validation loss | 用时 | 峰值 CUDA |
| --- | ---: | ---: | ---: | ---: | ---: |
| e1 | 48/48 | 1 / 12 | 1.2645 | 21.65 s | 5.249 GiB |
| e3 未平衡 | 48/48 | 3 / 36 | 0.8501 | 57.26 s | 5.273 GiB |
| e3、non-low x2 | 48/63 | 3 / 48 | 0.7220 | 80.84 s | 5.273 GiB |

冻结 20 样本结果：

| 模型状态 | 严格 JSON | 风险准确率 | 事件 micro-F1 | 主要现象 |
| --- | ---: | ---: | ---: | --- |
| e1 adapter | 45% | 10% | 0 | 11 条输出在 256 token 处截断 |
| e3 未平衡 adapter | 100% | 35% | 0 | 20 条全部预测为 low，发生类别塌缩 |
| e3、non-low x2 adapter | 100% | 50% | 0.182 | 17 low / 3 medium，恢复少量事件预测 |
| e3、non-low x2 merged | 100% | 45% | 0.100 | 合并后数值/生成结果与在线 adapter 不完全相同 |

最终 adapter 仅训练语言 attention 的 `q/k/v/o_proj`，可训练参数为 6422528，占
2133954560 总参数的 0.301%。产物校验：

| 产物 | SHA-256 |
| --- | --- |
| `adapter_model.safetensors` | `487b6a468a1372aa5bca610bfa7e1e508bd94b69d182a665019565ea6362886a` |
| merged `model.safetensors` | `9d12fa537bb34afd12b72a849fda0919d5df37c74fffd028f4919d78c3d496fe` |

这组数据说明 validation loss 下降不能替代冻结测试集质量评测。过采样缓解了全 low
塌缩，但事件识别仍弱于旧 LoRA；当前瓶颈首先是候选标签规模与质量，而不是继续增加
epoch。

### 28.5 领域 INT4 AWQ 量化与 ONNX 导出

新增 `scripts/quantize_qwen3_vl_int4_awq_domain.py`，将校准 JSONL 的 `text` 字段作为
TensorRT Edge-LLM `quantize_and_export` 的自定义 callable dataset，并在运行前验证
工具源码 commit。量化边界固定为：

- LLM backbone：W4A16 AWQ，group size 128；
- visual：FP16，不量化；
- `lm_head`：FP16，不量化；
- KV cache：不量化。

执行命令：

```bash
cd /root/autodl-tmp/JetsonVLM
export PATH=/root/autodl-tmp/TensorRT-Edge-LLM/.venv-int4/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/JetsonVLM/src

python scripts/quantize_model.py \
  --config configs/flows/quantize_qwen3_vl_2b_int4_awq_ps16_v1.json
python scripts/quantize_model.py \
  --config configs/flows/quantize_qwen3_vl_2b_int4_awq_ps16_v1.json \
  --execute

python scripts/export_model.py \
  --config configs/flows/export_qwen3_vl_2b_int4_awq_ps16_v1.json
python scripts/export_model.py \
  --config configs/flows/export_qwen3_vl_2b_int4_awq_ps16_v1.json \
  --execute
```

校准集身份与结果：

| 项目 | 结果 |
| --- | --- |
| calibration rows | 16 |
| calibration SHA-256 | `0949bfb7649f74a0a537781e5e46363d9b76cb3b046ecf4b91b6cd02171f77f3` |
| tokenizer 长度 min/max/mean | 416 / 449 / 425.81 |
| 超过 512 token | 0 |
| 量化耗时 | 约 69.0 s（内部量化阶段约 63.6 s） |
| 量化权重字节数 | 2185741496 |
| 量化权重 SHA-256 | `4d431087963e27a90040082366e32ceeeac22d6f4434eddbd08094ec99709d3b` |

导出的 LLM ONNX：

| 文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `model.onnx` | 4209515 | `78c7badb4cfa1f81dd71a66742b811c89131e232c7af6522fd73956ea59e9603` |
| `model.onnx.data` | 1350303744 | `2b44da3836ef2fb02c90a227d97cce36c4c197263d5b16579ee314c325fce712` |
| `embedding.safetensors` | 622329944 | `af9dbce9b96ca45a1cdc7c5ebdd18dc780567e707388e7cead78fbf1adb256d5` |

传输归档 `qwen3_vl_2b_int4_awq_ps16_v1_llm.tar` 为 1988280320 字节，SHA-256 为
`d99bdca5832bd7492bd6a8acde59841447bcf38cfbb640986f82a3518885e421`。

### 28.6 Jetson 传输、engine 构建与运行

服务器产物直接通过 SCP 传输到 Jetson，板端重新计算归档以及三个内部文件哈希，结果
与服务器全部一致。板端命令如下：

```bash
cd /home/ubuntu/JetsonVLM
mkdir -p artifacts/transfers artifacts/onnx/qwen3_vl_2b_int4_awq_ps16_v1
scp -P <AUTODL_PORT> <AUTODL_USER>@<AUTODL_HOST>:<REMOTE_ARCHIVE> \
  artifacts/transfers/qwen3_vl_2b_int4_awq_ps16_v1_llm.tar
sha256sum artifacts/transfers/qwen3_vl_2b_int4_awq_ps16_v1_llm.tar
tar -xf artifacts/transfers/qwen3_vl_2b_int4_awq_ps16_v1_llm.tar \
  -C artifacts/onnx/qwen3_vl_2b_int4_awq_ps16_v1

export PYTHONPATH=/home/ubuntu/JetsonVLM/src
.venv-jetson/bin/python scripts/build_engine.py \
  --config configs/flows/build_qwen3_vl_2b_int4_awq_ps16_v1_llm_engine_i768_k1024.json
.venv-jetson/bin/python scripts/build_engine.py \
  --config configs/flows/build_qwen3_vl_2b_int4_awq_ps16_v1_llm_engine_i768_k1024.json \
  --execute
```

构建在 `graphical.target` 下直接成功，用时约 41.75 秒。LLM engine 结果：

| 项目 | 结果 |
| --- | --- |
| profile | max input 768、KV capacity 1024 |
| engine 字节数 | 1362893996 |
| engine SHA-256 | `589d8ba247a93cdf794c86697bb5a5d5fe3387fee812744c51d09806912b3026` |
| visual engine | 复用相同基础模型 revision 的 FP16 visual engine |
| flow status | `succeeded` |

运行时与 study 命令：

```bash
cd /home/ubuntu/JetsonVLM
export PYTHONPATH=/home/ubuntu/JetsonVLM/src

PYTHONPATH=/home/ubuntu/JetsonVLM/src:/home/ubuntu/TensorRT-Edge-LLM \
EDGELLM_PLUGIN_PATH=/home/ubuntu/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so \
LD_LIBRARY_PATH=/home/ubuntu/JetsonVLM/.venv-jetson/lib/python3.10/site-packages/nvidia/cu12/lib:/home/ubuntu/TensorRT-Edge-LLM/build:/usr/local/cuda/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu \
.venv-jetson/bin/python scripts/serve_edgellm.py \
  --engine-root artifacts/engines/qwen3_vl_2b_int4_awq_ps16_v1_i768_k1024 \
  --host 127.0.0.1 --port 8000

curl http://127.0.0.1:8000/health
sudo tegrastats --interval 1000 \
  > reports/jetson-runtime-20260817-int4-domain/tegrastats_study.log 2>&1 &
PYTHONPATH=src .venv-jetson/bin/python -m parksight_vlm.app.run_study \
  --config configs/studies/jetson_edgellm_int4_awq_ps16_v1_ps20_pilot.json
```

`/health` 返回 healthy，20 条请求均获得 HTTP 200 并完成后端生成。评测结束后停止
runtime；Jetson 仍为 `graphical.target`，空闲时 `available` 内存恢复到约 5.3 GiB。

### 28.7 新领域 INT4 的冻结 20 样本结果

质量结果：

| 指标 | 新领域 ps16 INT4 | 旧通用 n128 INT4 |
| --- | ---: | ---: |
| 后端完成 | 20/20 | 20/20 |
| 严格 JSON 有效率 | 4/20（20%） | 20/20（100%） |
| 风险准确率 | 15% | 35% |
| 事件 micro-F1 | 0 | 0 |
| 失败 | 16 `json_parse_error` | 无 |

检查 16 条失败的 `raw_output` 后，主要模式为：

````text
```json
{...完整 JSON...}
```
````

即输出内容通常可人工识别为 JSON，但含有 Markdown 围栏，不满足项目规定的“仅输出
一个 JSON 对象”协议。评测器没有自动剥离围栏，因为这样会掩盖模型的格式遵循退化。

包含全部 20 次后端执行的性能摘要：

| 指标 | 结果 |
| --- | ---: |
| 端到端 min / mean | 9.26 / 11.04 s |
| 端到端 p50 / p90 / p99 | 10.68 / 12.60 / 12.60 s |
| 输出 token 总数 / 均值 | 1616 / 80.8 |
| 聚合输出速率 | 7.32 token/s |
| tegrastats 样本数 | 260 |
| RAM 均值 / 峰值 | 5347 / 5354 MB |
| swap | 969 / 12002 MB |
| GPU 利用率均值 / p50 / 峰值 | 81.88% / 99% / 99% |
| GPU 温度均值 / 峰值 | 59.75 / 62.97 C |
| 输入功耗均值 / 峰值 | 9.26 / 12.73 W |

旧通用 n128 INT4 的 p50 为 10.52 秒、聚合输出速率为 7.33 token/s。两版性能基本
相同，新版 RAM 峰值反而从约 5072 MB 增加到 5354 MB。因此当前 16 条领域文本校准
没有带来可确认的性能或质量收益。可能因素包括校准集仅一个 batch、领域覆盖窄、候选
答案不是人工金标，以及 W4A16 AWQ 对生成格式的敏感性；这些属于后续假设，不写成
已证实原因。

### 28.8 证据回传与当前状态

Jetson 证据已同步到本机忽略目录：

```text
reports/jetson-runtime-20260817-int4-domain/study.json
reports/jetson-runtime-20260817-int4-domain/summary.json
reports/jetson-runtime-20260817-int4-domain/server.log
reports/jetson-runtime-20260817-int4-domain/tegrastats_build.log
reports/jetson-runtime-20260817-int4-domain/tegrastats_study.log
reports/jetson-runtime-20260817-int4-domain/build-flow.json
reports/jetson-runtime-20260817-int4-domain/build-flow.log
```

服务器证据包已同步到：

```text
reports/server-sync-20260817/parksight_server_evidence_20260817.tar.gz
reports/server-sync-20260817/evidence/
```

证据包 SHA-256 为
`efff8bbb49a84cd1d7509ab029737b39de508c8ef60810e461b367907f0c682a`。包内保留 adapter、
训练指标、三轮 study、合并 study、量化 provenance 和 train/merge/quantize/export
flow；未重复回传 4.0 GiB 基础模型、2.0 GiB 量化权重或 ONNX 大文件。

本轮完成后，计划中的“用复核标签重训 LoRA”和“用独立领域校准集重做 INT4”都已有
真实运行证据。下一阶段不应盲目增加训练或量化轮次；如果继续提升质量，应先人工终审
标注并扩大 calibration 覆盖，再以相同冻结 20 样本复测。
