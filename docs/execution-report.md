# ParkSight-VLM 项目执行报告

更新时间：2026-07-28

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

当前已经完成：

- 泊车风险严格 JSON 领域对象、数据目录、冻结 workload 和来源组划分约束；
- Transformers 和 TensorRT Edge-LLM HTTP Runtime Adapter；
- 单图入口、配置化研究入口、质量/性能汇总和环境快照；
- LoRA 训练、合并、模型导出和 engine 构建的可审计流程包装器；
- Jetson Orin Nano 环境检查；
- `Qwen/Qwen3-VL-2B-Instruct` 指定 commit 的板端缓存和完整性校验；
- Jetson 专用 Python 3.10、PyTorch 2.9.1、torchvision 0.24.1 环境；
- `sm_87` CUDA kernel 验证；
- 一张真实泊车图片的多次 FP16 smoke test 及失败证据。

尚未完成：

- Qwen3-VL-2B 在当前板端成功生成严格 JSON；
- 冻结数据集、人工标注和完整 `StudyReport`；
- 服务器 Transformers 正确性参考；
- TensorRT Edge-LLM engine 构建与板端推理；
- LoRA 训练、合并复测和 INT4 对比。

当前阻塞不是 Python 接口、模型 revision 或 CUDA 架构不匹配，而是 L4T R36.4.7
运行期间出现的 `NvMapMemAllocInternalTagged ... error 12`。run5 关闭 PyTorch
caching allocator 后仍然失败。NVIDIA 论坛将同型问题列为 r36.4.7 已知问题，并说明
在 r36.5 修复：

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

## 15. 当前应做的下一步

### 15.1 建议

先备份板端关键环境，再将 L4T 从 R36.4.7 升级到包含 r36.5 的 JetPack 6.2.2，
重启后重复完全相同的 `sm_87` 验证和 run5。

系统升级和重启尚未执行。升级前需要单独确认：

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
冻结泊车数据集下载/整理
人工 reference assessment
完整 Jetson Transformers StudyReport
TensorRT Edge-LLM 模型导出
TensorRT Edge-LLM engine build
Jetson TensorRT Edge-LLM 推理
LoRA 训练和合并
INT4 LLM backbone
FP16/INT4 同机性能与质量对比
```

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
