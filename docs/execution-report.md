# ParkSight-VLM 项目执行报告

更新时间：2026-08-10

## 1. 报告说明

本文记录项目从早期仓库建立到当前 Jetson Transformers FP16 smoke test 的关键操作、
命令、结果和证据边界。命令使用以下标签：

- **已执行**：存在终端输出、提交、结构化结果或运行日志。
- **等价复现命令**：结果能够从仓库或文档确认，但原始终端的完整 quoting、参数顺序
  或输出没有保存；这里给出能够复现同一操作的等价命令。
- **尚未执行**：已经设计或提供入口，但没有实际结果。

本文不把代码入口、dry-run、模型文件存在或服务器结果写成 Jetson 部署完成。

## 2. 当前结论

项目已经实现以下软件链路：

```text
ParkingCase -> RiskRuntime -> InferenceRecord -> StudyReport
```

截至 2026-08-10 已经完成：

- 泊车风险严格 JSON 领域对象、数据目录、冻结 workload 和来源组划分约束；
- Transformers 和 TensorRT Edge-LLM HTTP Runtime Adapter；
- 单图入口、配置化研究入口、质量/性能汇总和环境快照；
- LoRA 训练、合并、模型导出和 engine 构建的可审计流程包装器；
- Jetson Orin Nano 环境检查；
- `Qwen/Qwen3-VL-2B-Instruct` 指定 commit 的板端缓存和完整性校验；
- Jetson 专用 Python 3.10、PyTorch 2.9.1、torchvision 0.24.1 环境；
- `sm_87` CUDA kernel 验证；
- 一张真实泊车图片的多次 Transformers FP16 smoke test 及失败证据；
- 服务器侧固定 revision 的 LLM 与视觉 ONNX 导出；
- Jetson 侧固定 commit 的 TensorRT Edge-LLM 编译；
- Jetson 侧 LLM/visual FP16 engine 构建；
- Edge-LLM 双 engine HTTP 服务、真实单图和冻结 20 样本同机评测；
- 20 样本端到端时延以及 RAM、swap、GPU、功耗和温度派生摘要。

尚未完成或未通过质量门槛：

- Qwen3-VL-2B 在当前板端生成通过严格 schema 的 JSON；
- 服务器 Transformers 正确性参考；
- Jetson Transformers FP16 在同一 20 样本上的成功 StudyReport；
- LoRA 训练、合并复测和 INT4 对比。

历史 Transformers 阻塞来自 L4T R36.4.7 期间的
`NvMapMemAllocInternalTagged ... error 12`；系统已升级到 R36.5.0。当前 Edge-LLM
部署链路已经运行，新的业务质量问题是 20/20 输出未通过严格 schema，而不是 engine
无法加载。

历史排查参考：

- <https://forums.developer.nvidia.com/t/intermittent-nvmapmemalloc-error-12-and-cuda-allocator-crash-during-pytorch-inference-on-jetson-orin-nano/349752>
- <https://forums.developer.nvidia.com/t/jetson-inference/365149/7>

## 3. 仓库演进

### 3.1 早期仓库阶段

以下历史只由 Git 提交证明，原始终端命令没有完整保存：

| 日期 | commit | 内容 |
| --- | --- | --- |
| 2026-07-01 | `14089ed` | 初始 Jetson Visual Memory Agent scaffold |
| 2026-07-05 | `e0eccf6` | 增加 embodied inference deployment workflow |
| 2026-07-17 | `1b59764` | 仓库重构 |
| 2026-07-20 | `66bb665` | 增加路线文档 |
| 2026-07-21 | `10346a0` | 重新定义仓库结构和目的 |

查看这些历史的等价命令：

```bash
git log --oneline --decorate --all
git show --stat 14089ed
git show --stat e0eccf6
git show --stat 1b59764
git show --stat 10346a0
```

早期 CLIP/SmolVLM 和 episode replay 代码属于仓库历史，不作为当前
Qwen3-VL + TensorRT Edge-LLM 项目的已完成部署证据。

### 3.2 ParkSight-VLM 主链路建立

2026-07-26 的 `bbb5715`：

```text
feat: establish ParkSight VLM study pipeline
73 files changed, 4039 insertions(+), 11 deletions(-)
```

主要实现：

- `ParkingAssessment` 严格字段、枚举和 schema 校验；
- `ParkingCase`、manifest、annotation、图片引用和来源组划分校验；
- `FrozenWorkload` 配置读取和 SHA-256 identity；
- `RiskRuntime`、`InferenceRecord` 和稳定失败类别；
- Qwen3-VL Transformers Adapter；
- TensorRT Edge-LLM OpenAI-compatible HTTP Adapter；
- `StudyRunner`、质量指标、性能分位数和环境快照；
- 单图命令、配置化研究命令；
- train/merge/export/build 四个显式流程包装器；
- 73 个文件对应的单元测试与 fixtures。

冻结工作负载：

```text
workload_id=parking_risk_v1
input_size=448x448
max_new_tokens=256
do_sample=false
```

初始无硬件验证采用的 Windows 等价命令：

```bat
set PYTHONPATH=src&python -m unittest discover -s tests
git diff --check
```

结果：

```text
Ran 27 tests
OK
git diff --check: passed
```

对应提交的等价命令：

```bash
git add AGENTS.md README.md CONTEXT.md configs data docs pyproject.toml scripts src tests
git commit -m "feat: establish ParkSight VLM study pipeline"
git push origin main
```

实际提交：

```text
bbb57157c9be93c71626334e2058284ebc067ac8
```

## 4. 实验角色确定

项目最终明确区分三种运行角色：

| 运行位置 | Runtime | 用途 |
| --- | --- | --- |
| GPU 服务器 | Transformers | 任务正确性参考、误差分析和训练 |
| Jetson | Transformers FP16 | 板端原生框架可运行性和性能基线 |
| Jetson | TensorRT Edge-LLM FP16/INT4 | 最终部署 Runtime |

服务器结果不用于证明 Jetson 性能收益。Jetson Transformers 与 TensorRT Edge-LLM
需要使用相同模型 commit、workload、数据集和板端功耗模式进行同机比较。

这一阶段只完成了代码和实验配置，没有执行服务器正确性参考。

## 5. Jetson 连接与系统环境

### 5.1 SSH

已确认地址：

```bash
ssh ubuntu@192.168.137.187
```

仓库位置：

```text
/home/ubuntu/JetsonVLM
```

### 5.2 设备、系统和资源检查

已执行的主要命令：

```bash
cat /proc/device-tree/model
cat /etc/nv_tegra_release
/usr/local/cuda/bin/nvcc --version
nvpmodel -q
free -h
swapon --show --bytes
df -h /
```

结果：

| 项目 | 结果 |
| --- | --- |
| 设备 | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super |
| GPU compute capability | `(8, 7)`，对应 `sm_87` |
| L4T | R36.4.7 |
| CUDA Toolkit | 12.6 |
| 功耗模式 | 15W，mode id 0 |
| RAM | 7.4 GiB |
| Swap | 3.7 GiB zram |
| NVMe | 233 GiB，总剩余约 181 GiB |

### 5.3 `free` 和 swap 调查

使用：

```bash
free -h
swapon --show --bytes
```

最终检查结果约为：

```text
Mem available: 5.7 GiB
Swap used: approximately 1 MiB
```

结论：

- `free` 列较小不能直接解释为内存泄漏，Linux 会把空闲 RAM 用作 page cache；
- 判断内存压力应同时查看 `available` 和实际 swap used；
- zram 处于 enabled 状态不等于系统正在大量换页；
- 当时最终快照没有显示持续内存压力。

现存证据没有保存完整的逐进程排查输出，因此本文不声称删除了某个具体服务或进程。

### 5.4 原 Python 环境

检查命令：

```bash
/home/ubuntu/project/llm-on-device/.venv/bin/python -V
/home/ubuntu/project/llm-on-device/.venv/bin/hf version
```

并通过该 Python 导入以下包查询版本：

```python
import torch
import transformers
import accelerate
import PIL
```

结果：

| 包 | 版本 |
| --- | --- |
| Python | 3.12.12 |
| torch | 2.9.1+cu126 |
| transformers | 4.57.6 |
| accelerate | 1.12.0 |
| Pillow | 12.1.0 |
| huggingface-hub CLI | 0.36.0 |

该 venv 后续确认是通用 aarch64 CUDA wheel，不是 Orin `sm_87` wheel。它被完整保留，
没有覆盖或卸载。

## 6. 模型 revision 与下载

### 6.1 不可变 commit

通过 `huggingface_hub.HfApi.model_info()` 解析官方模型仓库 SHA，最终固定：

```text
model_id=Qwen/Qwen3-VL-2B-Instruct
revision=89644892e4d85e24eaac8bacfd4f463576704203
```

不可变 commit 的作用是确保服务器 Transformers、Jetson Transformers、TensorRT
Edge-LLM、LoRA 和量化实验不会因 `main` 更新而静默使用不同权重。

### 6.2 Qwen3-0.6B 原缓存

原模型仓库缓存根目录：

```text
/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B
```

占用约 1.5 GiB，后续没有删除。

### 6.3 下载 Qwen3-VL-2B-Instruct

已执行：

```bash
/home/ubuntu/project/llm-on-device/.venv/bin/hf download \
  Qwen/Qwen3-VL-2B-Instruct \
  --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --cache-dir /home/ubuntu/.cache/huggingface/hub
```

最终 snapshot：

```text
/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
```

校验结果：

```text
files: 12/12
bytes: 4266648961/4266648961
.incomplete files: 0
model.safetensors SHA-256: matched Hugging Face LFS blob hash
cache size: approximately 4.0 GiB
```

板端 `huggingface_hub==0.36.0` 不支持：

```text
hf models info
hf download --dry-run
hf cache verify
```

因此使用 `HfApi.model_info(..., files_metadata=True)`、本地文件大小比较、
`.incomplete` 搜索和权重 SHA-256 完成等价校验，没有为了 CLI 子命令升级原 venv。

## 7. 仓库同步到 Jetson

PC 完成 push，用户在 Jetson 完成 pull。可确认的最终状态是板端仓库存在完整项目结构，
且后续运行位于：

```text
/home/ubuntu/JetsonVLM
```

原始 pull 终端输出没有保存。等价命令：

```bash
cd /home/ubuntu/JetsonVLM
git pull origin main
git status --short --branch
```

最终状态：

```text
## main...origin/main
```

## 8. 单图输入

用户将一张泊车图片放到板端：

```text
/home/ubuntu/JetsonVLM/data/raw/v2-5b48cc8372861680ce69be0d30d05319_1440w.jpg
```

`data/raw/` 已在 `.gitignore` 中，不进入 Git。

第一次命令没有输出的原因是 stdout 被重定向到结果文件，同时当时 CLI 在真正模型加载前
因未知参数退出。该事实推动了后续 CLI 修复。

## 9. 单图入口修复

### 9.1 精度参数没有传给 Transformers

原入口不接受：

```text
--device-map
--dtype
--attn-implementation
```

因此 `--dtype float16` 被 argparse 拒绝，结果文件为空，stderr 记录
`unrecognized arguments`。

commit `5e2bdfd` 完成：

- 增加三个 CLI 参数；
- 将它们放入 Transformers runtime options；
- 明确 `precision` 是实验身份，`dtype` 是实际加载精度；
- 增加对应测试。

提交：

```text
5e2bdfdd236c55e7281ae06dbf9595f542dfc442
fix: pass precision options to single-image runtime
```

### 9.2 CUDA 架构预检

run1 后确认原 torch：

```text
torch=2.9.1+cu126
device=Orin
capability=(8, 7)
arch_list=['sm_80', 'sm_90']
```

64 MiB CUDA tensor 返回：

```text
no kernel image is available for execution on the device
```

commit `731fa2f` 在模型加载前检查：

```python
required_architecture = f"sm_{major}{minor}"
supported_architectures = torch.cuda.get_arch_list()
```

如果缺少 `sm_87`，生成稳定的 `dependency_unavailable`，避免把底层 allocator 文本误判
成模型容量问题。

提交：

```text
731fa2f7d8507e1432ac07d1bcb98f2b3a20d9cd
fix: detect incompatible Jetson CUDA wheels
```

## 10. 原 Python 3.12 环境的 run1/run2

### 10.1 run1

使用 `5e2bdfd`、模型 commit、FP16、`device_map=auto` 和 SDPA。原始 shell quoting
没有完整保留，等价推理命令如下：

```bash
cd /home/ubuntu/JetsonVLM
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src \
timeout 900s \
/home/ubuntu/project/llm-on-device/.venv/bin/python \
  -m parksight_vlm.app.analyze_image \
  --image data/raw/v2-5b48cc8372861680ce69be0d30d05319_1440w.jpg \
  --runtime transformers \
  --backend-revision transformers==4.57.6 \
  --model-revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --precision fp16 \
  --device-map auto \
  --dtype float16 \
  --attn-implementation sdpa
```

结果：

```text
exit code: 2
failure category: runtime_error
end-to-end failure time: approximately 15.25 s
tegrastats RAM peak: approximately 4.0/7.6 GiB
GR3D_FREQ: 0%
stderr: NvMap error 12 followed by PyTorch NVML allocator assertion
```

由于该 torch 不包含 `sm_87`，run1 不能作为有效 FP16 容量测试。

### 10.2 run2

应用 `731fa2f` 后重跑，约 7.09 秒返回：

```text
category=dependency_unavailable
installed torch 2.9.1+cu126 does not include CUDA kernels for sm_87;
supported architectures: sm_80, sm_90
```

stderr 为空，说明失败由项目结构化记录。

### 10.3 文档提交

run1/run2 和阻塞写入：

```text
f362a4397a2bfd58b7367201c4f26d2b3fc3dd0e
docs: record Jetson FP16 smoke failures
```

截至该阶段已 push 的提交：

```text
5e2bdfd
731fa2f
f362a43
```

## 11. Jetson Python 3.10 隔离环境

### 11.1 venv 创建

首先尝试：

```bash
python3.10 -m venv /home/ubuntu/JetsonVLM/.venv-jetson
```

结果：

```text
failed: ensurepip/python3.10-venv unavailable
```

随后使用板端已有 uv：

```bash
/home/ubuntu/.local/bin/uv venv \
  --clear \
  --seed \
  --no-project \
  --python /usr/bin/python3.10 \
  /home/ubuntu/JetsonVLM/.venv-jetson
```

结果：

```text
Python 3.10.12
pip 26.1.2
setuptools, wheel and packaging seeded
```

该环境位于仓库内但被 `.gitignore` 排除，不覆盖原 Python 3.12 venv。

### 11.2 Jetson PyTorch wheel

选择：

```text
torch-2.9.1-cp310-cp310-linux_aarch64.whl
```

不可变下载 URL：

```text
https://pypi.jetson-ai-lab.io/jp6/cu126/+f/02f/de421eabbf626/torch-2.9.1-cp310-cp310-linux_aarch64.whl
```

SHA-256：

```text
02fde421eabbf62633092de30405ea4d917323c55bea22bfd10dfeb1f1023506
```

板端直接下载速度只有约 24 MiB/27 分钟，因此停止该下载。PC 不使用代理时也很慢，
随后通过 PC 已配置代理下载完整 wheel。下面是等价复现命令：

```powershell
curl.exe -L --fail `
  --proxy http://127.0.0.1:7897 `
  --output "$env:TEMP\torch-2.9.1-cp310-cp310-linux_aarch64.whl" `
  "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/02f/de421eabbf626/torch-2.9.1-cp310-cp310-linux_aarch64.whl"

Get-FileHash `
  -Algorithm SHA256 `
  -LiteralPath "$env:TEMP\torch-2.9.1-cp310-cp310-linux_aarch64.whl"
```

实际校验：

```text
bytes=228271497
SHA256=02fde421eabbf62633092de30405ea4d917323c55bea22bfd10dfeb1f1023506
match=True
```

传输到板端的等价命令：

```powershell
scp "$env:TEMP\torch-2.9.1-cp310-cp310-linux_aarch64.whl" `
  ubuntu@192.168.137.187:/home/ubuntu/JetsonVLM/.venv-jetson/
```

板端再次运行 `sha256sum`，结果一致。

安装：

```bash
/home/ubuntu/.local/bin/uv pip install \
  --python /home/ubuntu/JetsonVLM/.venv-jetson/bin/python \
  /home/ubuntu/JetsonVLM/.venv-jetson/torch-2.9.1-cp310-cp310-linux_aarch64.whl
```

torch 依赖中没有通用 `nvidia-cuda-*` wheel，只安装 filelock、fsspec、jinja2、
networkx、sympy、typing-extensions 等 Python 依赖。

### 11.3 Transformers 依赖

安装：

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

结果：

```text
transformers==4.57.6
accelerate==1.12.0
Pillow==12.1.0
huggingface-hub==0.36.0
```

### 11.4 cuDSS

首次 `import torch`：

```text
ImportError: libcudss.so.0: cannot open shared object file
```

`ldd libtorch_cuda.so` 只发现这一项缺失库。指定 NVIDIA cuDSS aarch64 修正版：

```bash
/home/ubuntu/.local/bin/uv pip install \
  --no-deps \
  --python /home/ubuntu/JetsonVLM/.venv-jetson/bin/python \
  'nvidia-cudss-cu12==0.4.0.2.post1'
```

使用 `--no-deps` 是为了复用 JetPack 的 CUDA 12.6/cuBLAS，不让依赖解析安装整套通用
CUDA 12.9 Python wheel。

库位置：

```text
/home/ubuntu/JetsonVLM/.venv-jetson/lib/python3.10/site-packages/nvidia/cu12/lib/libcudss.so.0
```

运行新环境时需要：

```bash
export LD_LIBRARY_PATH=/home/ubuntu/JetsonVLM/.venv-jetson/lib/python3.10/site-packages/nvidia/cu12/lib:/usr/local/cuda/lib64
```

### 11.5 torchvision

run3 暴露：

```text
AutoVideoProcessor requires the Torchvision library
```

`torch 2.9.1` 对应 Jetson `torchvision 0.24.1`：

```bash
/home/ubuntu/.local/bin/uv pip install \
  --python /home/ubuntu/JetsonVLM/.venv-jetson/bin/python \
  'https://pypi.jetson-ai-lab.io/jp6/cu126/+f/d5b/caaf709f11750/torchvision-0.24.1-cp310-cp310-linux_aarch64.whl'
```

结果：

```text
torch=2.9.1
torchvision=0.24.1
cuda=True
```

## 12. `sm_87` CUDA 验证

使用新环境执行 CUDA tensor：

```bash
export LD_LIBRARY_PATH=/home/ubuntu/JetsonVLM/.venv-jetson/lib/python3.10/site-packages/nvidia/cu12/lib:/usr/local/cuda/lib64

/home/ubuntu/JetsonVLM/.venv-jetson/bin/python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"arch_list={torch.cuda.get_arch_list()}")
print(f"device={torch.cuda.get_device_name(0)}")
print(f"capability={torch.cuda.get_device_capability(0)}")
x = torch.ones(1024 * 1024, device="cuda", dtype=torch.float16)
y = (x * 3).sum()
torch.cuda.synchronize()
print(f"kernel_result={y.item()}")
PY
```

结果：

```text
torch=2.9.1
cuda_available=True
cuda_runtime=12.6
arch_list=['sm_87']
device=Orin
capability=(8, 7)
kernel_result=inf
```

`kernel_result=inf` 是 1048576 个 FP16 数求和超过 FP16 最大有限值，不是 kernel
失败；CUDA kernel 已实际执行并同步完成。

新环境仓库测试：

```bash
cd /home/ubuntu/JetsonVLM
LD_LIBRARY_PATH=/home/ubuntu/JetsonVLM/.venv-jetson/lib/python3.10/site-packages/nvidia/cu12/lib:/usr/local/cuda/lib64 \
PYTHONPATH=src \
.venv-jetson/bin/python -m unittest discover -s tests
```

结果：

```text
Ran 29 tests in 0.037s
OK
```

## 13. 新环境真实图片 smoke test

每次推理同时运行：

```bash
nohup tegrastats \
  --interval 1000 \
  --logfile reports/smoke/RUN_NAME_tegrastats.log \
  </dev/null >/dev/null 2>&1 &
echo $!
```

推理通用形式：

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
  --attn-implementation sdpa \
  > reports/smoke/RUN_NAME.json \
  2> reports/smoke/RUN_NAME_stderr.log
```

完成后只终止刚才输出的 tegrastats PID：

```bash
kill TEGRSTATS_PID
```

### 13.1 run3

身份：

```text
jetson_transformers_fp16_20260728_run3
```

结果：

| 项目 | 值 |
| --- | --- |
| failure | `runtime_error / ImportError` |
| end-to-end | 12257.90 ms |
| tegrastats RAM peak | 1962 MiB |
| 最小 lfb | 3 × 4 MiB |
| 最大 GR3D | 0% |
| 最大 VDD_IN | 5793 mW |
| 最大温度 | 50.781 C |

原因：

```text
AutoVideoProcessor requires torchvision
```

该问题随后通过安装 Jetson torchvision 0.24.1 解决。

证据：

- `reports/smoke/jetson_transformers_fp16_20260728_run3.json`
- `reports/smoke/jetson_transformers_fp16_20260728_run3_stderr.log`
- `reports/smoke/jetson_transformers_fp16_20260728_run3_tegrastats.log`

### 13.2 run4

身份：

```text
jetson_transformers_fp16_20260728_run4
```

结果：

| 项目 | 值 |
| --- | --- |
| failure | `runtime_error / RuntimeError` |
| end-to-end | 12150.17 ms |
| tegrastats RAM peak | 4473 MiB |
| 最小 lfb | 2 × 4 MiB |
| 最大 GR3D | 0% |
| 最大 VDD_IN | 5832 mW |
| 最大温度 | 50.937 C |

stderr：

```text
NvMapMemAllocInternalTagged: 1075072515 error 12
NvMapMemHandleAlloc: error 0
NVML_SUCCESS == r INTERNAL ASSERT FAILED
```

失败发生在权重加载阶段，尚未进入 GPU 计算。

### 13.3 run5

为了判断是否为 PyTorch caching allocator 的大块预分配，增加：

```bash
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
```

PyTorch 官方说明该变量会关闭 CUDA allocation caching：

<https://docs.pytorch.org/docs/main/cuda_environment_variables.html>

结果：

| 项目 | 值 |
| --- | --- |
| failure | `out_of_memory / AcceleratorError` |
| end-to-end | 10525.44 ms |
| tegrastats RAM peak | 4005 MiB |
| 最小 lfb | 2 × 4 MiB |
| 最大 GR3D | 0% |
| 最大 VDD_IN | 5793 mW |
| 最大温度 | 51.125 C |
| swap | 0 |

进程退出后的系统状态：

```text
Mem used: 1.2 GiB
Mem free: 3.6 GiB
Mem available: 5.9 GiB
Swap used: 0
```

这说明报错不能简单解释为“7.4 GiB 总 RAM 已经用完”。总可用内存仍较多，但
`lfb` 只剩 2 个 4 MiB 连续块，底层 NvMap 大块分配返回 error 12。结合
R36.4.7 的同型已知问题，当前不能据此得出“Qwen3-VL-2B FP16 必然无法放入 8GB”的
结论。

证据：

- `reports/smoke/jetson_transformers_fp16_20260728_run5.json`
- `reports/smoke/jetson_transformers_fp16_20260728_run5_stderr.log`
- `reports/smoke/jetson_transformers_fp16_20260728_run5_tegrastats.log`

## 14. 清理与最终状态

本轮完成后：

- 停止所有本轮 tegrastats；
- 停止 PC 上超时后仍运行的 torch wheel 下载进程；
- 删除 PC 的完整/不完整临时 wheel；
- 删除板端传输用 wheel，已安装的 venv 保留；
- 保留全部 run3–run5 JSON、stderr 和 tegrastats；
- PC 与 Jetson Git 都是 `main...origin/main`；
- 没有产生新的已跟踪代码修改；
- 原 Python 3.12 venv 未修改。

run3–run5 原始证据已复制到本机 `reports/smoke/`。该目录由 `.gitignore` 忽略，
不会因普通 commit 进入仓库。

## 15. 2026-07-28 当时的下一步

### 15.1 建议

先备份板端关键环境，再将 L4T 从 R36.4.7 升级到包含 r36.5 的 JetPack 6.2.2，
重启后重复完全相同的 `sm_87` 验证和 run5。

以下为当时的升级前检查项；升级与重启后来已经完成，结果见第 19 节：

- 当前启动介质和 JetPack 安装方式；
- NVIDIA apt source；
- 可回滚备份；
- 电源稳定；
- 升级包范围；
- 升级后 L4T、CUDA、cuDNN、TensorRT 兼容性。

### 15.2 升级后验收顺序

1. `cat /etc/nv_tegra_release`
2. `free -h` 和 `swapon --show`
3. Python 3.10 venv `import torch`
4. `torch.cuda.get_arch_list()` 包含 `sm_87`
5. 小 tensor CUDA kernel
6. 29 个无硬件单元测试
7. 单图 run6 + tegrastats
8. 如果成功，再建立冻结数据集

## 16. 尚未执行的项目操作

以下内容已有代码、配置或设计，但没有实际结果：

```text
服务器 Qwen3-VL Transformers 正确性参考
完整 Jetson Transformers StudyReport
LoRA 训练和合并
INT4 LLM backbone
FP16/INT4 同机性能与质量对比
```

TensorRT Edge-LLM 模型导出已于 2026-08-04 完成，实际证据见第 18 节；Jetson engine
构建与真实推理已于 2026-08-10 完成，见第 19–20 节。

## 17. 后续命令协作约定

后续默认流程：

1. Codex 给出一组小范围命令；
2. 说明命令目的；
3. 给出预期输出和判断标准；
4. 用户亲自执行；
5. 用户贴回完整输出；
6. Codex解释结果并给下一步命令。

除非用户在当前请求明确要求 Codex 代执行，否则 Codex 不再直接 SSH、安装依赖、
下载模型、运行 GPU 任务、升级系统或执行 Git commit/push。

## 18. TensorRT Edge-LLM 服务器 ONNX 导出（2026-08-04）

### 18.1 固定身份与环境

| 项目 | 实测值 |
| --- | --- |
| 服务器系统 | Ubuntu 22.04.5 LTS，x86-64 |
| GPU | NVIDIA GeForce RTX 4090 D，24 GiB |
| 驱动 / `nvidia-smi` CUDA | 595.71.05 / 13.2 |
| 导出 Python | 3.12.3，独立 `.venv-export` |
| PyTorch | `2.12.0+cu126`，CUDA 可用 |
| TensorRT Edge-LLM | `0.9.1`，commit `7f061f21f0a581ba234a1e233c9315b89d8e47d6` |
| Transformers / ONNX / ONNX Script | `5.9.0` / `1.19.0` / `0.7.0` |
| 模型 revision | `89644892e4d85e24eaac8bacfd4f463576704203` |

服务器没有系统 `nvcc`，但固定 PyTorch CUDA 12.6 runtime 的 FP16 matrix smoke test
通过，TensorRT Edge-LLM 的 Python ONNX 导出不依赖本次服务器 C++ 编译。

### 18.2 模型与流程输入校验

镜像下载后的单文件权重实际信息为：

```text
model.safetensors bytes  = 4255140312
model.safetensors sha256 = 7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0
```

该 revision 发布单个 `model.safetensors`，不发布
`model.safetensors.index.json`。因此导出 flow 的 `required_inputs` 已改为实际单文件
权重，并增加回归断言，防止配置再次要求不存在的 index 文件。

### 18.3 执行命令与结果

在服务器 `JetsonVLM` 仓库执行：

```bash
export PYTHONPATH=/root/autodl-tmp/JetsonVLM/src
export PATH=/root/autodl-tmp/TensorRT-Edge-LLM/.venv-export/bin:$PATH

python scripts/export_model.py \
  --config configs/flows/export_qwen3_vl_2b_fp16.json

python scripts/export_model.py \
  --config configs/flows/export_qwen3_vl_2b_fp16.json \
  --execute
```

dry-run 返回 `ready: true`、`missing_inputs: []`、`preexisting_outputs: []`。
实际执行从 `2026-08-04T15:12:59.286487+00:00` 到
`2026-08-04T15:15:14.059600+00:00`，退出码为 0，record 状态为
`succeeded`。LLM 与视觉编码器的全部声明输出均为 `true`，ONNX 目录约 4.6 GiB。

本机保存的证据为：

```text
reports/flows/export_qwen3_vl_2b_fp16.json
reports/flows/export_qwen3_vl_2b_fp16.log
artifacts/transfers/qwen3_vl_2b_fp16_export_89644892.tar
```

归档长度为 `4890972160` 字节，服务器与本机合并归档的 SHA-256 均为：

```text
7e00cd92099ff9f35ed600e68682630b70c87cc58e1311874a15002d63fc4e45
```

归档内 11 个 ONNX 与 sidecar 文件的独立 SHA-256 已全部复算为 `OK`。该证据只证明
x86 服务器导出成功；Jetson runtime 编译、engine 构建和推理仍需在目标板端验证。

## 19. Jetson TensorRT Edge-LLM 编译与 engine 构建（2026-08-10）

### 19.1 板端环境与 ONNX 交接

板端已升级并实测为 L4T R36.5.0、JetPack 6.2.2、CUDA 12.6、TensorRT 10.3。
服务器导出的归档在板端解包后，11 个文件的 SHA-256 全部通过；归档身份仍为：

```text
qwen3_vl_2b_fp16_export_89644892.tar
bytes  = 4890972160
sha256 = 7e00cd92099ff9f35ed600e68682630b70c87cc58e1311874a15002d63fc4e45
```

Jetson Python 环境为 `.venv-jetson` / Python 3.10.12，PyTorch 2.9.1、Transformers
4.57.6、Accelerate 1.12.0。cuDSS 动态库已随虚拟环境存在，但启动 Python 时需要将
其目录放入 `LD_LIBRARY_PATH`；设置后 `torch.cuda.is_available()` 为 `True`，arch
list 为 `['sm_87']`：

```bash
export JETSON_PY_CUDA_LIB=$PWD/.venv-jetson/lib/python3.10/site-packages/nvidia/cu12/lib
export LD_LIBRARY_PATH=$JETSON_PY_CUDA_LIB:$LD_LIBRARY_PATH
```

### 19.2 固定源码编译

TensorRT Edge-LLM 固定为 `v0.9.1` commit
`7f061f21f0a581ba234a1e233c9315b89d8e47d6`。板端离线补齐 NVTX、googletest 与
nlohmann/json 子模块，使用项目 `.venv-jetson` 中的 CMake 3.31.6 和 pybind11 2.13.6
完成编译。最终验证存在：

```text
build/libNvInfer_edgellm_plugin.so
build/examples/llm/llm_build
build/examples/llm/llm_inference
build/examples/multimodal/visual_build
build/pybind/_edgellm_runtime.cpython-310-aarch64-linux-gnu.so
```

`ldd` 未发现缺失依赖，`_edgellm_runtime.LLMRuntime` 和
`experimental.server.engine.LLM` 均可导入。为兼容 JetPack 6.2.2 / TensorRT 10.3，
仓库保存了 `patches/tensorrt-edge-llm/` 下的显式补丁，不修改上游固定 commit 身份。

### 19.3 FMHA CUBIN 调查

LLM ONNX 第一个 `AttentionPlugin` 最初因 `CUDA_ERROR_INVALID_IMAGE` 创建失败。逐条
验证 11 个 SM87 FMHA CUBIN 后，9 个可加载，只有两个 `head_size=256 + custom_mask`
资产被 CUDA 驱动拒绝；目标 Qwen3-VL-2B 配置为 `head_size=128`。插件补丁仅在驱动
明确返回 `INVALID_IMAGE` 时记录警告并跳过该资产，随后 32 层 AttentionPlugin 均能
创建，ONNX 解析和 TensorRT 图优化可继续。

### 19.4 LLM 构建期内存证据

已归档六次失败 record/log，文件名从
`build_qwen3_vl_2b_fp16_engines.attempt1_*.{json,log}` 到
`attempt6_weight_streaming_oom.{json,log}`。依次排除了插件路径、CUBIN、默认优化级别、
workspace 和 weight streaming 开关问题。内核记录显示 `llm_build` 在 8 GiB 统一内存
和约 3.7 GiB zram 用尽后被 OOM killer 终止；另一次 TensorRT 明确记录额外
`3441150208` 字节分配失败。

为降低搜索峰值，独立 LLM flow 固定：

```text
maxBatchSize              = 1
maxInputLen               = 1024
maxKVCacheCapacity        = 2048
workspace limit           = 1024 MiB
builder optimization      = 0
weight streaming          = enabled
```

板端已创建但尚未启用 `/home/ubuntu/parksight-build.swap`，大小 8 GiB、权限 0600，
swap UUID 为 `ef2f7643-de66-4127-8c11-f7f3d2574e17`。启用需要用户在板端输入 sudo
密码：

```bash
sudo swapon --priority 1 /home/ubuntu/parksight-build.swap
swapon --show
free -h
```

### 19.5 视觉 engine 成功证据

为避免 LLM OOM 覆盖视觉阶段，构建入口已支持 `--component llm|visual|both`，并增加
两个独立 flow。视觉 flow
`configs/flows/build_qwen3_vl_2b_fp16_visual_engine.json` 实际执行成功：

```text
plan_identity = 769d2168534bfcbed1cf34c673fabdbe5d0b3314193059251306255412c78793
status        = succeeded
build time    = 33.991 s
TRT GPU peak  = 789 MiB
TRT CPU peak  = 4417 MiB
```

输出证据：

```text
artifacts/engines/qwen3_vl_2b_fp16/visual/visual.engine
bytes  = 824000540
sha256 = 3c6b4cce682e021b09c066d0e325335e31ef9edbf613c754be586035c26f5c2f

reports/flows/build_qwen3_vl_2b_fp16_visual_engine.json
reports/flows/build_qwen3_vl_2b_fp16_visual_engine.log
```

截至第 19.5 节记录时，该证据只证明视觉 engine 构建；随后完成的 LLM engine、双
engine 加载、真实单图和 20 样本结果见第 20 节。

## 20. Jetson FP16 engine 与真实推理验收（2026-08-10）

### 20.1 临时 swap 与 LLM engine

`/home/ubuntu/parksight-build.swap` 已实际启用。重复执行 `swapon` 时出现：

```text
insecure file owner 1000, 0 (root) suggested
swapon failed: Device or resource busy
```

`/proc/swaps` 和 `swapon --show` 证明该 8 GiB 文件已经处于 active 状态；第二行表示
重复启用，不是 swap 失效。文件未写入 `/etc/fstab`。

随后执行独立 LLM flow：

```bash
cd /home/ubuntu/JetsonVLM
PYTHONPATH=src .venv-jetson/bin/python scripts/build_engine.py \
  --config configs/flows/build_qwen3_vl_2b_fp16_llm_engine.json \
  --execute
```

结果：

```text
flow_id       = build_qwen3_vl_2b_fp16_llm_7f061f21
plan_identity = d3266941b1ed48b75a09a3ef13971e668089f69ee975b6adda049fcf642f2314
status        = succeeded
started_at    = 2026-08-10T00:36:48.769548+00:00
finished_at   = 2026-08-10T00:40:37.732676+00:00
```

输出：

```text
artifacts/engines/qwen3_vl_2b_fp16/llm/llm.engine
bytes  = 3453798316
sha256 = cbdf0300bf406dfbbcd06d47435c699c26403139d6bdd06b473ba00576583013
```

### 20.2 运行时 weight streaming 补丁

第一次启动服务时，默认相对插件路径落到了 ParkSight 工作目录，导致
`AttentionPlugin` 未注册。显式设置绝对 `EDGELLM_PLUGIN_PATH` 后，engine 能完成
反序列化，但 TensorRT 默认关闭 weight streaming，并尝试一次性分配
`3441150208` 字节 GPU 权重，随后 OOM。

项目因此增加
`patches/tensorrt-edge-llm/0009-configurable-runtime-weight-streaming-budget.patch`，
在创建 `IExecutionContext` 前读取
`EDGELLM_WEIGHT_STREAMING_BUDGET_BYTES` 并调用
`ICudaEngine::setWeightStreamingBudgetV2()`。板端单线程增量编译：

```bash
cd /home/ubuntu/TensorRT-Edge-LLM
/home/ubuntu/JetsonVLM/.venv-jetson/bin/cmake --build build \
  --target _edgellm_runtime --parallel 1
```

启动命令的关键参数为：

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

日志确认：

```text
requested=0 actual=0 streamable=3441150208 scratch=1244660224 bytes
Base EngineExecutor successfully loaded
Vision runner successfully initialized
Setup shared execution context memory: 1356027904 bytes
Successfully captured decoding CUDA graphs
Uvicorn running on http://127.0.0.1:8000
```

`GET /health` 返回 HTTP 200 和 `status=healthy`。可选 action engine 不存在只产生信息
日志，不影响本项目的视觉语言链路。

### 20.3 真实单图

执行：

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

后端返回 HTTP 200、57 个 token，端到端 `38943.90 ms`。模型生成了可解析 JSON，
但把 `events`、`evidence` 和 `driver_advice` 输出为字符串，因此严格领域解析返回：

```text
category       = json_parse_error
message        = events must be an array
exception_type = AssessmentValidationError
```

这证明真实视觉与生成链路已执行，但不构成业务 schema 成功。

### 20.4 冻结 20 样本同机评测

执行：

```bash
cd /home/ubuntu/JetsonVLM
tegrastats --interval 1000 \
  > reports/runtime/jetson_edgellm_fp16_ps20_pilot.tegrastats.log &

PYTHONPATH=src .venv-jetson/bin/python -m parksight_vlm.app.run_study \
  --config configs/studies/jetson_edgellm_fp16_ps20_pilot.json
```

服务日志记录 21 个成功 POST，其中 1 个为单图、20 个为 study。StudyReport 包含完整
20 条记录，stderr 为空：

```text
sample_count                 = 20
backend_completed            = 20
strict schema valid          = 0
failure_summary              = {"json_parse_error": 20}
end-to-end p50/p90/p99       = 41.76/50.61/55.42 s
aggregate output throughput  = 1.48 token/s
```

874 条 tegrastats 的派生结果：

```text
RAM peak                     = 7414 MB
swap peak                    = 2579 MB
GPU utilization mean        = 98.79%
VDD_IN mean/max              = 10.11/10.76 W
GPU temperature mean/max     = 60.67/62.50 C
```

`scripts/summarize_jetson_study.py` 从原始 StudyReport 与 tegrastats 生成派生摘要，
明确把“后端完成”与“严格 schema 成功”分开。原始与派生证据保存在本机忽略目录：

```text
reports/jetson-runtime-20260810/
  build_qwen3_vl_2b_fp16_llm_engine.json
  build_qwen3_vl_2b_fp16_llm_engine.log
  edgellm_fp16_server_run3.log
  jetson_edgellm_fp16_ps20-indoor-001.json
  jetson_edgellm_fp16_ps20_pilot.json
  jetson_edgellm_fp16_ps20_pilot.tegrastats.log
  jetson_edgellm_fp16_ps20_pilot.runtime-summary.json
```

### 20.5 LoRA 与 INT4 阶段记录

当前只有 20 个冻结 `test` 样本，没有独立 `train`/`validation` 数据。LoRA 示例配置
仍引用不存在的 `data/manifests/parking_risk_v1.jsonl` 与对应 annotation，flow command
仍为 `replace-with-reviewed-training-command`。因此本轮没有执行训练、没有 adapter、
没有合并模型，也没有把 test 集用于训练。

INT4 当前同样只有规划，没有校准数据、量化配置、INT4 engine 或板端报告。FP16
weight streaming 的 3.44 GB streamable weights、1.24 GB scratch、约 7.4 GB RAM 占用
和 41.76 秒 p50 已构成后续 INT4 的客观动机，但不能代替 INT4 实测证据。

## 21. Prompt 与统一内存修复后的最终验收（2026-08-12）

### 21.1 修复内容

- 将 Qwen3-VL 的 system message 固定为 content 数组结构，并用真实 chat template
  fixture 覆盖 Transformers 与 Edge-LLM 两条 Adapter。
- 实测 Edge-LLM C++ 最终请求长度为 735 token，将 LLM engine profile 固定为
  `maxInputLen=768`、`maxKVCacheCapacity=1024`。
- 在固定 Edge-LLM commit 上应用
  `patches/tensorrt-edge-llm/preallocate-base-context-before-multimodal.patch`，先申请
  1,347,639,296 字节 base/decoder execution context，再加载 visual runner，解决 8 GB
  统一内存上加载顺序导致的 `cudaMalloc` OOM。

最终 LLM engine 为 `3453786212` 字节，SHA-256 为
`5a679dceb7fdbe5661dd5ae67ba28804f0e18341f10adeb99b2906c2d8cfcf05`。补丁后的
Python runtime binding 为 `47559536` 字节，SHA-256 为
`cc9518f4adbe9a6e2cf7512b9cf9803e80d6491d09bf8d8a46092b966d2ab4de`。

### 21.2 单图与 20 样本结果

同一图片连续三次均返回 HTTP 200、`failure=null` 和通过严格 schema 的 assessment，
平均端到端时延 `76.19 s`，平均速度 `1.496 token/s`。

随后以独立 study id 运行冻结 `ps20_pilot_v1`：

```text
backend completed       = 20/20
strict JSON valid       = 20/20
failure summary         = {}
risk level accuracy     = 0.35
event micro-F1          = 0.3590
unsafe advice rate      = 0
end-to-end p50/p90/p99  = 50.75/69.08/75.08 s
aggregate output rate   = 1.480 token/s
```

542 条 tegrastats 的派生结果：

```text
RAM peak                = 7418 MB
swap peak               = 1904 MB
GPU utilization mean   = 97.39%
VDD_IN mean/max         = 10.05/10.70 W
GPU temperature mean/max = 63.13/65.03 C
```

该结果把 2026-08-10 的严格 JSON 有效率从 0/20 修复到 20/20，并证明补丁后的 runtime
能够承受完整连续负载。风险准确率与事件 F1 仍偏低，因此 LoRA 是质量优化阶段，不再是
修复部署或格式链路的前置条件。原始证据与派生摘要保存在：

```text
reports/jetson-runtime-20260812/
```
