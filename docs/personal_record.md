# JetsonVLM项目学习记录

## 1. Jetson板端环境检查/依赖安装/版本更新

### 1.1 板端环境检查

查看开发套件版本：

```bash
cat /proc/device-tree/model
```

```text
NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
```

查看 Jetson 系统版本：

```bash
cat /etc/nv_tegra_release
```

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```

查看cuda版本和nvcc版本：

```bash
/usr/local/cuda/bin/nvcc --version
```

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Wed_Aug_14_10:14:07_PDT_2024
Cuda compilation tools, release 12.6, V12.6.68
Build cuda_12.6.r12.6/compiler.34714021_0
```

查看JetPack版本和L4T版本：

```bash
apt-cache policy nvidia-jetpack nvidia-l4t-core
```

```text
nvidia-jetpack:
  已安装：6.2.1+b38
  候选： 6.2.1+b38
  版本列表：
 *** 6.2.1+b38 600
        600 https://repo.download.nvidia.com/jetson/common r36.4/main arm64 Packages
        100 /var/lib/dpkg/status
     6.2+b77 600
        600 https://repo.download.nvidia.com/jetson/common r36.4/main arm64 Packages
     6.1+b123 600
        600 https://repo.download.nvidia.com/jetson/common r36.4/main arm64 Packages
nvidia-l4t-core:
  已安装：36.4.7-20250918154033
  候选： 36.4.7-20250918154033
  版本列表：
 *** 36.4.7-20250918154033 600
        600 https://repo.download.nvidia.com/jetson/t234 r36.4/main arm64 Packages
        100 /var/lib/dpkg/status
     36.4.4-20250616085344 600
        600 https://repo.download.nvidia.com/jetson/t234 r36.4/main arm64 Packages
     36.4.3-20250107174145 600
        600 https://repo.download.nvidia.com/jetson/t234 r36.4/main arm64 Packages
     36.4.0-20240912212859 600
        600 https://repo.download.nvidia.com/jetson/t234 r36.4/main arm64 Packages
```

查看功率模式：

```bash
nvpmodel -q
```

```text
NV Power Mode: 15W
0
```

查看系统target：

```bash
systemctl get-default
```

```bash
graphical.target
```

查看硬件相关资源：

```bash
free -h
```

```text
               total        used        free      shared  buff/cache   available
Mem:           7.4Gi       1.5Gi       4.6Gi        45Mi       1.4Gi       5.7Gi
Swap:          3.7Gi          0B       3.7Gi
```

```bash
swapon --show --bytes
```

```text
NAME       TYPE           SIZE USED PRIO
/dev/zram0 partition 665829376    0    5
/dev/zram1 partition 665829376    0    5
/dev/zram2 partition 665829376    0    5
/dev/zram3 partition 665829376    0    5
/dev/zram4 partition 665829376    0    5
/dev/zram5 partition 665829376    0    5
```

**Swap（交换空间）** 是指系统将存储空间（如固态硬盘或内存压缩区）模拟当作**虚拟内存**使用的一种技术。当物理内存（RAM）快要用完时，系统会把不常用的数据暂时移到 Swap 里面，从而防止程序因为内存不足而崩溃。

```bash
df -h
```

```text
Filesystem       Size  Used Avail Use% Mounted on
/dev/nvme0n1p1   233G   41G  181G  19% /
tmpfs            3.8G  172K  3.8G   1% /dev/shm
tmpfs            1.5G   43M  1.5G   3% /run
tmpfs            5.0M  4.0K  5.0M   1% /run/lock
/dev/nvme0n1p10   63M  110K   63M   1% /boot/efi
tmpfs            762M  120K  762M   1% /run/user/1000
```

Jetson 板端专用Pytorch环境检查：

板端torch不能通过常规方式下载，需通过Jetson PyTorch wheel下载。

```bash
cd ~/JetsonVLM
.venv-jetson/bin/python -m pip list
```

```text
Package            Version
------------------ -------------
accelerate         1.12.0
certifi            2026.7.22
charset-normalizer 3.4.9
filelock           3.32.0
fsspec             2026.6.0
hf-xet             1.5.2
huggingface-hub    0.36.0
idna               3.18
Jinja2             3.1.6
MarkupSafe         3.0.3
mpmath             1.3.0
networkx           3.4.2
numpy              2.2.6
nvidia-cudss-cu12  0.4.0.2.post1
packaging          26.2
pillow             12.1.0
pip                26.1.2
psutil             7.2.2
PyYAML             6.0.3
regex              2026.7.19
requests           2.34.2
safetensors        0.8.0
setuptools         83.0.0
sympy              1.14.0
tensorrt           10.3.0
tensorrt_dispatch  10.3.0
tensorrt_lean      10.3.0
tokenizers         0.22.2
torch              2.9.1
torchvision        0.24.1
tqdm               4.70.0
transformers       4.57.6
typing_extensions  4.16.0
urllib3            2.7.0
wheel              0.47.0
```

其中，Transformer相关依赖通过板端UV下载。

### 1.2 更新JetPack和L4T

第一次运行Transformer时发现当前 torch 来自 PyTorch CUDA 12.6 通用`cp312-manylinux_2_28_aarch64` wheel。它能发现 Orin，但不包含 Orin 所需的 `sm_87`。目标 JetPack 6.2.2 对应 L4T R36.5，必须先切换软件源。

```bash
# 备份软件源
sudo cp -a /etc/apt/sources.list.d/nvidia-l4t-apt-source.list /etc/apt/sources.list.d/nvidia-l4t-apt-source.list.pre-r36.5
# 将36.4改为36.5
sudo sed -i 's/r36\.4/r36.5/g' /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
# 确认结果，预期输出：common/t234/ffmpeg r36.5 main
cat /etc/apt/sources.list.d/nvidia-l4t-apt-source.list 
# 更新apt
sudo apt update
# 智能升级软件包
sudo apt dist-upgrade
```

## 2. 项目记录

### 2.1 评测链路搭建

评测链路总览：

```text
ParkingCase -> RiskRuntime -> InferenceRecord -> StudyReport
```

**模型输出、人工标注、评测标准使用同一套 JSON 契约（ParkingAssessment）**。该字段为模型输出 / 人工标注共用契约。

| 字段 | 类型 | 含义 | 校验规则 |
| --- | --- | --- | --- |
| `schema_version` | string | schema 版本标识 | 必须等于 `parking_risk_v1` |
| `risk_level` | string 枚举 | 总体风险等级 | 只能是 `low` / `medium` / `high` |
| `events` | string 数组 | 识别出的风险事件（可空） | 必须数组、元素在 6 个枚举内、不重复 |
| `evidence` | string 数组 | 图像证据（可见线索） | 必须非空数组、不重复 |
| `driver_advice` | string 数组 | 驾驶建议 | 必须非空数组、元素在 5 个枚举内、不重复 |

6 个风险事件枚举：`vru_near_maneuver_path`（行人接近路径）/ `vehicle_near_maneuver_path`（车辆接近路径）/
`fixed_obstacle_near_path`（固定障碍物）/ `narrow_passage`（狭窄通道）/ `visibility_occlusion`（可见性遮挡）/
`parking_space_conflict`（车位冲突）。

5 个驾驶建议枚举：`maintain_observation`（保持观察）/ `slow_down`（减速）/ `yield`（让行）/
`prepare_to_stop`（准备停车）/ `change_maneuver_when_safe`（安全时改变操作）。

#### 数据文件 JSON 格式

**manifest**（样本清单，`data/manifests/ps20_pilot_v1.jsonl`），每行一个样本：

```json
{"case_id":"ps20-indoor-001","image_ref":"raw/ps2.0/pilot/indoor/001.jpg","source_group_id":"ps2.0-testing-indoor","split":"test"}
```

| 字段 | 含义 |
| --- | --- |
| `case_id` | 样本唯一 ID |
| `image_ref` | 图片相对 `data/` 的路径 |
| `source_group_id` | 来源组（防泄漏划分） |
| `split` | 数据集划分：train / validation / test |

**annotation**（人工标注，`data/annotations/ps20_pilot_v1.jsonl`）：

```json
{"case_id":"ps20-indoor-001","assessment":{"schema_version":"parking_risk_v1","risk_level":"medium","events":["narrow_passage"],"evidence":["自车左右两侧均有近距离停放车辆，可见横向通行空间较窄。"],"driver_advice":["slow_down","maintain_observation"]}}
```

`RiskRuntime.analyze()` 执行一个样本后产出，关键约束：`assessment` 与 `failure` **恰好二选一**。

主要字段：`case_id` / `runtime_identity`（backend、model_id、model_revision、adapter_revision、precision）/
`workload_identity`（SHA-256）/ `assessment` | `failure` / `raw_output`（模型原始输出，留痕复盘）/
`stage_timings`（preprocess、model_generate、end_to_end 等，未测为 null）/ `resource_snapshot`（峰值显存）/ `output_tokens`。

失败分类（`_classify_failure`）：`json_parse_error` / `dependency_unavailable` / `out_of_memory` /
`timeout` / `model_refusal` / `unsupported_operator` / `runtime_error` / `input_error`。

`StudyRunner.run()` 聚合所有记录：

| 字段 | 含义 |
| --- | --- |
| `study_identity` | 实验完整身份（study_id + workload + runtime + split + 功耗模式） |
| `environment_snapshot` | 环境快照（L4T、Python、torch/transformers 版本） |
| `quality_metrics` | 质量指标（对照人工标注） |
| `performance_metrics` | 性能指标（**只统计成功记录**） |
| `failure_summary` | 失败类别计数 |
| `records` | 全部原始 InferenceRecord，一条不删 |

质量指标：`json_validity_rate`（JSON 有效率）/ `risk_level_accuracy`（风险等级准确率）/
`event_micro_f1`（事件 micro-F1，六类事件累计 TP/FP/FN）/ `unsafe_advice_rate`（不安全建议率）/
`event_errors`（逐事件 FP/FN）。

性能指标：`cold_start_ms`（冷启动 = 第一条成功记录端到端）/ `stage_latency_ms`（每阶段 p50/p90/p99）/
`tokens_per_second` / `peak_memory_mb` / `average_power_w` / `peak_temperature_c`。

#### 评测标准

本项目的评测标准如下：

| 概念             | 含义                     | 本项目的例子                                      |
| ---------------- | ------------------------ | ------------------------------------------------- |
| **TP（真阳性）** | 标注有事件，模型也预测有 | 标注有 `narrow_passage`，模型也标了 ✅             |
| **FP（假阳性）** | 标注没有，模型却预测有   | 标注没有，模型标了 `narrow_passage` ❌（过度报警） |
| **FN（假阴性）** | 标注有，模型漏掉了       | 标注有 `narrow_passage`，模型没标 ❌（漏报）       |

```
precision = TP / (TP + FP)   预测为有的事件里，有多少是真的
recall    = TP / (TP + FN)   真有的事件里，模型找回了多少
F1        = 2·P·R / (P+R)    两者的调和平均
```

最终使用`micro-F1`，即对所有类别的TP/FP/FN累加，最后计算得到。

#### 运行前测试

随后冷测试run1-run6，属于基本链路测试，其中遇到了不少问题，如格式不支持fp16、未下载Torchvision包、sm不匹配、JetPack版本错误等。

| run      | 环境 / 修改                        | 结果                                | 得到的结论                                   |
| -------- | ---------------------------------- | ----------------------------------- | -------------------------------------------- |
| 入口初测 | 原 CLI                             | `--dtype` 等参数无法识别            | 增加参数并传入 runtime options               |
| run1     | 原 Python 3.12 通用 torch          | NvMap error 12，约 15.25 s          | 此结果混有错误 torch wheel，不能判断模型容量 |
| run2     | 增加 CUDA arch 预检                | `dependency_unavailable`，约 7.09 s | 明确发现 torch 不含 `sm_87`                  |
| run3     | 新 `.venv-jetson`                  | 缺少 torchvision，12.26 s           | 安装 Jetson torchvision 0.24.1               |
| run4     | 依赖补齐                           | NvMap/NVML allocator error，12.15 s | 权重加载阶段连续大块分配失败                 |
| run5     | `PYTORCH_NO_CUDA_MEMORY_CACHING=1` | `out_of_memory`，10.53 s            | 不是单纯 caching allocator 预留导致          |

升级到R36.5.0之后：

| run  | 结果                                             | 原因 / 修复                                                  |
| ---- | ------------------------------------------------ | ------------------------------------------------------------ |
| run6 | `runtime_error: string indices must be integers` | Qwen3-VL 多模态 system message 结构不符合 processor 期望     |
| run7 | 模型已生成 JSON，但严格解析失败                  | `events` 使用中文自由文本，`evidence`/`driver_advice` 还是字符串 |
| run8 | `failure=null`，严格 JSON 成功                   | 修复消息格式并加强 prompt 的英文 snake_case 枚举约束         |

几个关键设计：

- 懒加载：`_ensure_loaded()`只在首次`generate`时才加载模型，CLI启动、配置校验、无硬件测试都不需要模型权重
- `_require_cuda_architecture`进行CUDA架构预检

```python
required = f"sm_{major}{minor}"   # Orin Nano → sm_87
if required not in torch.cuda.get_arch_list():
    probe = torch.ones(1, device="cuda"); probe.add_(1)  # 真实 kernel 试跑
```

这里发现了通用aarch64 torch wheel只包含`sm_80/sm_90`，而实际需要`sm_87`。

- 冻结workload：`max_new_tokens=256`、`do_sample=False`（贪心解码，可复现）、输入 448×448、prompt 严格约束输出 JSON 枚举等

### 2.2 板端基线FP16 VL Transformer推理实现

#### 数据集准备

面向泊车场景，主要调研了三个数据集：

| 数据集 | 描述 | 限制 |
| --- | --- | --- |
| WoodScape | 真实车载四向鱼眼，人、车、障碍、遮挡 | 不专注停车场，数据许可为 proprietary |
| Tongji PS2.0 | 四鱼眼拼接的 AVM 鸟瞰图、停车位 | 缺少通用障碍和风险标签 |
| nuScenes | 包含大量汽车在城市道路和停车场进行倒车、掉头、泊车时的同步后视图像 | 不专注停车场景 |

使用[Tongji PS2.0](https://cslinzhang.github.io/deepps/)数据集，并选取20pilot样本进行手动标注，结果见[对应jsonl文件](../data/annotations/ps20_pilot_v1.jsonl)。

#### HuggingFace模型下载

通过SHA下载固定commit版本的模型，保证服务器端、Jetson Transformer、Jetson TensorRT模型权重相同。

```bash
/home/ubuntu/project/llm-on-device/.venv/bin/hf download \
  Qwen/Qwen3-VL-2B-Instruct \
  --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --cache-dir /home/ubuntu/.cache/huggingface/hub
```

结果如下：

- 模型：`Qwen/Qwen3-VL-2B-Instruct`
- 固定 commit：`89644892e4d85e24eaac8bacfd4f463576704203`
- snapshot：
  `/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203`
- 仓库缓存占用：约 4.0 GiB

##### Qwen3-VL-2B-Instruct模型结构

Qwen3-VL 是**混合架构 VLM**：一个视觉塔（ViT）+ 一个 LLM 骨干，中间通过视觉-文本对齐层融合。总参数 **21.34 亿**（项目实测 `2133954560`）。

```text
                     ┌─────────────────────────────────────┐
                     │          Qwen3-VL-2B 整体            │
                     │                                     │
 图片 ──► 视觉塔(ViT) ──► 视觉 token ──┐                    │
             24 层                     │  融合               │
             hidden 1024               ├─► LLM 骨干 (28 层) ─► 文本输出
             patch 16×16               │  hidden 2048        │
                                      └─► 文本 token ──┘     │
                     └─────────────────────────────────────┘
```

视觉塔（Vision Tower）—— `vision_config`

| 参数                       | 值        | 含义                                       |
| -------------------------- | --------- | ------------------------------------------ |
| `depth`                    | 24        | 24 层 Transformer                          |
| `hidden_size`              | 1024      | 每层隐藏维度                               |
| `num_heads`                | 16        | 注意力头数                                 |
| `patch_size`               | 16        | 图片切成 16×16 patch                       |
| `temporal_patch_size`      | 2         | 视频时间维采样（本项目只用单图）           |
| `spatial_merge_size`       | 2         | **视觉 token 合并**（下面重点讲）          |
| `out_hidden_size`          | 2048      | 输出维度，对齐到 LLM 的 2048               |
| `deepstack_visual_indexes` | [5,11,17] | 从第 5/11/17 层取**分层特征**（DeepStack） |

LLM 骨干

| 参数                      | 值     | 含义                                                         |
| ------------------------- | ------ | ------------------------------------------------------------ |
| `num_hidden_layers`       | 28     | **28 层** decoder（项目实测：LoRA 注入"28 层 self-attention"；32 个 AttentionPlugin 是含视觉等插件的总数） |
| `hidden_size`             | 2048   | 隐藏维度                                                     |
| `intermediate_size`       | 6144   | FFN 中间维度（3× hidden）                                    |
| `num_attention_heads`     | 16     | Q 头数                                                       |
| `num_key_value_heads`     | 8      | **KV 头数 = 8（GQA）**                                       |
| `head_dim`                | 128    | 每头维度（项目实测 FMHA 用 head_size=128 的关键！）          |
| `vocab_size`              | 151936 | 词表                                                         |
| `max_position_embeddings` | 262144 | 极长上下文支持                                               |
| `tie_word_embeddings`     | true   | **embedding 与 lm_head 共享权重**                            |
| `attention_bias`          | false  | 注意力无 bias                                                |

#### 板端推理

`transformers.py`的`_ensure_loaded()` 加载对应模型和固定的commit：

```python
self._processor = AutoProcessor.from_pretrained(
    self._model_id,              # "Qwen/Qwen3-VL-2B-Instruct"
    revision=self._model_revision # 固定 commit，命中本地 snapshot
)
self._model = Qwen3VLForConditionalGeneration.from_pretrained(
    self._model_id,
    revision=self._model_revision,
    device_map="auto",           # 整模型映射到 cuda:0（实测 hf_device_map = {'': 0}）
    dtype="float16",             # FP16 精度
    attn_implementation="sdpa",  # SDPA 注意力省显存
)
```

配置`jetson_transformers_fp16_ps20_pilot.json`的`runtime`字段，固定模型信息：

```json
"runtime": {
  "backend": "transformers",
  "backend_revision": "transformers==4.57.6",
  "model_id": "Qwen/Qwen3-VL-2B-Instruct",
  "model_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
  "adapter_revision": "none",
  "precision": "fp16",
  "options": {
    "device_map": "auto",
    "dtype": "float16",
    "attn_implementation": "sdpa"
  }
}
```

执行推理：

```bash
# 冻结20样本study
cd /home/ubuntu/JetsonVLM

LD_LIBRARY_PATH=/home/ubuntu/JetsonVLM/.venv-jetson/lib/python3.10/site-packages/nvidia/cu12/lib:/usr/local/cuda/lib64 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src \
timeout 1800s \
.venv-jetson/bin/python -m parksight_vlm.app.run_study \
  --config configs/studies/jetson_transformers_fp16_ps20_pilot.json
```

运行结果结果如下：

| 指标 | 结果 |
| --- | ---: |
| 样本数 | 20 |
| JSON 有效率 | 1.000 |
| 风险等级准确率 | 0.350 |
| 事件 micro precision | 0.280 |
| 事件 micro recall | 0.4375 |
| 事件 micro F1 | 0.3415 |
| 不安全建议率 | 0.000 |
| 运行时失败 | 0 |

主要误差：20 张全部预测为 `low`；18 张预测 `narrow_passage`，7 张额外预测`vehicle_near_maneuver_path`，没有预测 `visibility_occlusion`。这表明 JSON 契约已经成立，但业务判断质量较低。

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
| `VDD_IN` 最终区间平均 | 10.115 W |

### 2.3 TensorRT-edge-llm部署

#### GPU服务器导出TensorRT Edge-LLM ONNX

选择[[AutoDL算力云](https://www.autodl.com/market/list)]平台的单卡RTX 4090 D作为服务器端，具体环境如下：

| 项目                              | 实测值                                                    |
| --------------------------------- | --------------------------------------------------------- |
| 系统                              | Ubuntu 22.04.5 LTS，x86-64                                |
| GPU                               | NVIDIA GeForce RTX 4090 D，24 GiB                         |
| 驱动 / `nvidia-smi` CUDA          | 595.71.05 / 13.2                                          |
| 导出 Python                       | 3.12.3，独立 `.venv-export`                               |
| PyTorch                           | `2.12.0+cu126`                                            |
| TensorRT Edge-LLM                 | v0.9.1，commit `7f061f21f0a581ba234a1e233c9315b89d8e47d6` |
| Transformers / ONNX / ONNX Script | 5.9.0 / 1.19.0 / 0.7.0                                    |

```bash
# 固定源码和模型
cd /root/autodl-tmp
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git checkout --detach 7f061f21f0a581ba234a1e233c9315b89d8e47d6 # 该revision是单个model.safetensors
git submodule update --init --recursive
git rev-parse HEAD
# dry-run与导出
export PYTHONPATH=/root/autodl-tmp/JetsonVLM/src
export PATH=/root/autodl-tmp/TensorRT-Edge-LLM/.venv-export/bin:$PATH

python scripts/export_model.py \
  --config configs/flows/export_qwen3_vl_2b_fp16.json

python scripts/export_model.py \
  --config configs/flows/export_qwen3_vl_2b_fp16.json \
  --execute
```

导出耗时约为3分钟，输出包含 LLM 和视觉编码器的 ONNX、external data、tokenizer/chat template、视觉preprocessor/config 等共 11 个文件。

```text
llm/
  model.onnx                    ← LLM 主干图结构（小，几 MB）
  model.onnx.data               ← LLM 权重 external data（大，~3.4 GB）★
  config.json                   ← LLM 配置
  embedding.safetensors         ← embedding 权重单独导出（~620 MB）
  tokenizer.json                ← 分词器
  tokenizer_config.json         ← 分词器配置
  processed_chat_template.json  ← 已处理的 chat 模板（engine 推理时用）
visual/
  model.onnx                    ← 视觉编码器图结构（小）
  model.onnx.data               ← 视觉编码器权重 external data（~786 MB）
  config.json                   ← 视觉配置
  preprocessor_config.json      ← 视觉预处理配置（归一化/resize 参数）
```

Qwen3-VL-2B-Instruct的总参数量为21.34亿，FP16导出大小约为4.27 GB。

#### Jetson编译TensorRT Edge-LLM

**engine 与 GPU 架构、TensorRT 版本、插件、构建参数强绑定**，因此编译构建需要在板端进行。

```bash
# Jetson板端， Edge-LLM v0.9.1
cd /home/ubuntu
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git checkout --detach 7f061f21f0a581ba234a1e233c9315b89d8e47d6
git submodule update --init --recursive
git rev-parse HEAD   # 必须输出 7f061f21f0a581ba234a1e233c9315b89d8e47d6
```

仓库上游依赖NVTX、googletest、nlohmann/json三个子模块，从PC传输；CMake和Pybind11从`.venv-jeston`中安装。

通过CMake构建和编译：

```bash
cd /home/ubuntu/TensorRT-Edge-LLM
mkdir -p build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin \
  -DCUDA_CTK_VERSION=12.6 \
  -DENABLE_CUTE_DSL=ALL \
  -DBUILD_PYTHON_BINDINGS=ON \
  -Dpybind11_DIR="$PYBIND11_DIR"

cmake --build . --parallel 2   # 低并行度！避免 8GB 板端编译 OOM
```

编译产物如下：

```text
build/libNvInfer_edgellm_plugin.so              	← TensorRT 插件（AttentionPlugin 等）
build/examples/llm/llm_build                    	← LLM engine 构建器
build/examples/llm/llm_inference                 	← LLM 推理示例
build/examples/multimodal/visual_build           	← 视觉 engine 构建器
build/pybind/_edgellm_runtime.cpython-310-aarch64-linux-gnu.so  ← Python binding
```

根据板端实际环境，以补丁形式完成兼容性修复，如TensorRT 新旧版接口兼容问题、CuteDSL补丁。

##### FMHA cubin问题

***\*FMHA cubin\**** 是指针对 **FlashMulti-Head Attention（融合多头注意力机制）** 编译好的 **CUDA 二进制目标代码文件**（.cubin）。FlashAttention 的融合实现——把 attention 的 QK^T、softmax、PV 融合进一个 kernel，避免中间矩阵写回显存。**CUBIN**是CUDA 编译出的**二进制 kernel 文件**（类似 .exe 但给 GPU 执行）。Edge-LLM 针对每个 GPU 架构（这里是 sm_87）预先编译了多种 attention 配置的 CUBIN（比如不同 head_size、是否带 mask），运行时按模型配置挑一个加载。

遇到的问题是LLM ONNX 图优化到11AttentionPlugin时发现有2个创建失败，看 CUBIN 文件名（`*sm87.cubin.cpp` 是 C 数组包装）和元数据，发现被拒的两个都是：

```text
head_size = 256 + custom_mask
```

而Qwen3-VL-2B的实际attention配置是：

```text
head_size = 128
```

修复方法：跳过这两个CUBIN。

#### 启动TensortRT 推理服务

##### 运行时OOM问题修复

Jetson 是**统一内存架构**（UMA），CPU 和 GPU 共享同一块物理内存（8 GB），没有独立显存。

**weight streaming问题**：

TensorRT 的 **weight streaming** 机制：默认所有权重一次性常驻 GPU；开启后，权重可以**按需从内存流式加载**（用到哪层加载哪层），只保留一部分在 GPU 上。

探测实验发现：

```text
streamable weights              = 3441150208 bytes  ← 权重总量 3.44 GB
budget=0        → scratch = 1244660224 bytes（1.16 GiB 工作区）
budget=256MiB..3GiB → scratch 不降，总占用反而随 budget 涨
budget=全量(3.44GB) → scratch 归零，但常驻权重申请直接 OOM
```

因此只能选择budget=0，代价是权重流式读取让性能比 Transformers 还慢 4.45×。

**分配顺序问题**：

1.35 GB context + 0.79 GB visual 要同时存在。

假设先加载visual engine权重（0.79GB），再一次性申请LLM共享context，这时内存已碎，1.35GB连续内存申请失败

修正申请顺序：

```text
① 先查询 base executor / decoder strategy 的 context 大小
② 先申请并绑定 LLM 共享 GPU context (~1.35 GB)   ← 趁内存还完整时拿大块
③ 再加载 visual/audio/action runner (~0.79 GB)
④ 仅当 multimodal runner 需求确实更大时才重新分配
```

**engine profile问题**：

原始profile大小为`maxInputLen=1024`，`maxKVCacheCapacity=2048`，运行时KV Cache过大导致OOM，通过分析单次输入长度：

```text
Python processor: input_ids = 570 token（图片 196 visual tokens + 文本）
Edge-LLM C++ 实际展开 = 735 token（比 Python 多，模板展开不同）
```

将profile大小改为768/1024。

##### 正常启动服务

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

##### Transformers 与Edge-LLM同机比较

两份报告共同使用：

```text
model revision    89644892e4d85e24eaac8bacfd4f463576704203
workload identity parking_risk_v1@sha256:8350ace4574f8aa154319f7136ef831003d4dcc074ef20b74c1b419d69a2a493
dataset           ps20_pilot_v1 / test / 20 samples
power mode        15W_MODE_0
Jetson L4T        R36.5.0
precision         fp16
```

| 指标           | Transformers FP16 | Edge-LLM FP16 |
| -------------- | ----------------: | ------------: |
| 后端完成       |             20/20 |         20/20 |
| 严格 JSON 有效 |             20/20 |          0/20 |
| E2E p50        |           9.385 s |      41.758 s |
| E2E p90        |          14.117 s |      50.612 s |
| E2E p99        |          27.939 s |      55.419 s |
| RAM 峰值       |           6398 MB |       7414 MB |
| swap 峰值      |            501 MB |       2579 MB |
| 平均输入功耗   |          10.115 W |      10.110 W |

按当前数据计算，Edge-LLM 的 p50/p90/p99 分别约为 Transformers 的`4.45× / 3.59× / 1.98×`，即当前实现没有加速，反而更慢。主要背景是 LLM engine
必须使用 `weight-streaming-budget=0` 才能在 8 GB 设备上加载，权重流式访问和 1.24 GBscratch 带来明显代价。同时，两边模型权重相同，但 Edge-LLM 的 prompt/chattemplate、请求格式、解码行为或输出约束仍可能存在适配差异，导致严格JSON有效率极低。

### 2.4 量化

Jetson Orin + JetPack 6.2 支持 **FP16 / INT8 / INT4**，不支持 FP8/NVFP4。

#### 量化方法

使用W4A16的AWQ量化方法，配置如下：

```json
{
  "quant_algo": "W4A16_AWQ",   // 权重 W=INT4，激活 A=FP16
  "group_size": 128,           // 每 128 个权重共享一组 scale/zero-point
  "zero_point": false,         // 对称量化
  "pre_quant_scale": true,
  "KV cache quant": "none",
  "excluded": ["lm_head", "model.visual*"]  // ← 不量化这些
}
```

Qwen3-VL 里**只量化 `text_config` 对应的 LLM 骨干**（28 层 Transformer），具体是：

- 每层的 `q_proj / k_proj / v_proj / o_proj`（attention 投影）
- 每层的 `gate_proj / up_proj / down_proj`（FFN 投影）

AWQ共分为两步：

```text
# 第一步：weight-only量化
原始 FP16 权重 W (shape [out, in])
  → 按 group_size=128 分组
  → 每组：scale = max(|W_group|) / 8（INT4 对称范围 [-8, 7]）
  → W_int4 = round(W / scale)    每个权重存 4 bit
  → 反量化：W_approx = W_int4 × scale（推理时）
# 第二步：activation-aware 的 scale 优化
用校准集跑一遍，统计每个通道的激活值分布（activation 重要性）
→ 找到"对输出影响大"的权重通道（激活值大的通道）
→ 对这些通道放大 scale（per-channel 保护），减少量化误差
→ 把放大吸收进相邻层（等效变换，数学上无损）
```

#### 校准集准备

**为什么需要校准集**：AWQ 的"activation-aware"——它要跑一遍数据统计**每个权重通道对输出的影响**，才能决定量化时保护哪些通道。没有校准集就没有感知，量化就退化成纯 round。

使用通用新闻文本 `cnn_dailymail` 前 128 条作为校准集。

#### 环境准备与服务器端量化

```text
# RTX 4090 D 环境确认
torch 2.12.0 (CUDA 13), torchvision 0.27.0, transformers 5.9.0
ModelOpt 0.44.0（NVIDIA 量化工具）, datasets 4.8.5, ONNX 1.19
```

```bash
python scripts/quantize_model.py \
  --config configs/flows/quantize_qwen3_vl_2b_int4_awq.json \
  --execute
```

实测结果如下：

```text
flow status         succeeded
flow elapsed        113.50 s
ModelOpt quantization 98.4 s
峰值显存            约 8.5 GiB
```

AWQ内部做了什么：

1. 对 LLM 骨干每个 Linear 权重（q/k/v/o + FFN 的 gate/up/down）按 **group_size=128** 分组
2. 每 128 个权重算一组 scale：`scale = max(|W_group|) / 8`（INT4 对称范围 [-8,7]）
3. **跑校准集**统计激活值分布 → 找出对输出影响大的通道 → **放大这些通道的 scale（per-channel 保护）**，并把放大等效吸收进相邻层（数学无损变换）
4. 权重存 INT4，scale/zero 元数据保存

#### ONNX导出与板端构建

```bash
python scripts/export_model.py \
  --config configs/flows/export_qwen3_vl_2b_int4_awq.json \
  --execute
```

耗时98.19s，导出：

```text
llm/model.onnx              4209515 字节  (~4 MB)      图结构
llm/model.onnx.data         1350303744 字节  (~1.26 GiB)  INT4 权重（external data）
llm/embedding.safetensors   622329944 字节  (~594 MB)   embedding 单独存
```

会同时导出FP16 visual engine，但与板端完全一致。将上述文件传输至板端后构建engine：

```bash
.venv-jetson/bin/python scripts/build_engine.py \
  --config configs/flows/build_qwen3_vl_2b_int4_awq_llm_engine_i768_k1024.json --execute
```

#### INT4量化模型板端推理

执行串行单图稳定性验证，然后冻结 20 样本 INT4 Study。与FP 16模型对比：

| 指标               | Edge-LLM FP16 | Edge-LLM INT4 AWQ |       变化 |
| ------------------ | ------------: | ----------------: | ---------: |
| engine 大小        |  3453786212 B |      1362769140 B |     -60.5% |
| mean 延迟          |   53638.78 ms |       10685.66 ms | 5.02x 加速 |
| p50 延迟           |   50753.15 ms |       10524.38 ms | 4.82x 加速 |
| p90 延迟           |   69082.40 ms |       11394.11 ms | 6.06x 加速 |
| p99 延迟           |   75081.88 ms |       12626.09 ms | 5.95x 加速 |
| aggregate tokens/s |        1.4803 |            7.3276 |      4.95x |
| RAM peak           |      7418 MiB |          5072 MiB |     -31.6% |
| swap peak          |      1904 MiB |           840 MiB |     -55.9% |
| GPU 温度 peak      |      65.03 °C |          60.75 °C |   -4.28 °C |
| JSON validity      |          1.00 |              1.00 |       持平 |
| risk accuracy      |          0.35 |              0.35 |       持平 |
| event micro-F1     |        0.3590 |            0.0000 |   明显退化 |
| unsafe advice rate |          0.00 |              0.00 |       持平 |

从结果上看，量化后的模型在延迟、吞吐、engine 大小和内存上获得明确收益。但 128 条通用新闻文本的校准集只覆盖语言统计，不覆盖泊车视觉指令分布；本实验的事件 F1 退化说明该校准方案只能作为部署与性能验证，不能作为最终质量版本。

### 2.5 LoRA

从2.3节的Transformer推理结果和Edge-LLM的同精度推理结果中可以看出，两者针对”泊车场景“的准确率均很低，即便是Transformer模型，其”风险等级准确率“也仅有0.35。因此，需要对模型进行训练与微调。

#### 数据准备

##### 隔绝训练/验证/测试集

从 PS2.0 `training` 9827 张按**文件名提取来源组**（`p2_img28_0408` → `p2_img28`，同一采集序列归一组），每组只选一张：

```text
train  64 张 / 64 来源组
val    16 张 / 16 来源组
test   20 张 / 独立 pilot 组
```

保证`train-val overlap=0`和`train-test overlap=0`。

##### 弱监督标签生成

用固定基础模型 + 固定 workload 对 80 张图生成弱监督 JSON，teacher模型本身具有偏置，因此本轮只能证明”链路可行“。

#### 训练

只微调语言模块的q/k/v/o，因为”风险判断“的语义推理发生在LLM的attention。

```text
# 训练脚本
precision: BF16（训练省显存）
gradient_checkpointing: 开启（省显存）
gradient_accumulation: 4
AdamW, lr=1e-4, warmup 10%, weight decay 0
epochs=1, optimizer steps=16（64 样本 / 4 累积 = 16 步）
loss 除以 accumulation → 等效 batch=4
```

#### 评测

#### 参数量计算

A/B的低维度均为16。

Qwen3-VL-2B 的 `hidden_size=2048`，但注意 **GQA**：`num_attention_heads=16`、`num_key_value_heads=8`、`head_dim=128`：

```
q_proj: 2048 → 2048   (16 头)    参数 = 16×(2048+2048) = 65,536
k_proj: 2048 → 1024   (8 KV 头)  参数 = 16×(2048+1024) = 49,152
v_proj: 2048 → 1024   (8 KV 头)  参数 = 16×(2048+1024) = 49,152
o_proj: 2048 → 2048   (16 头)    参数 = 16×(2048+2048) = 65,536
─────────────────────────────────────────────
每层小计                                229,376
```

共28层decoder layer，总参数为：`229376 * 28 = 6,422,528`，约占总参数量的`6,422,528 / 2,133,954,560 = 0.3010%`。

### 2.6 2026-08-17：候选标注、数据拆分与 Transformers profiling

前面的 LoRA 记录基于弱监督标签和 64/16 的训练验证拆分。后续实验发现，80 条teacher 标签的 `risk_level` 全部为 `low`，其中 65 条包含 `narrow_passage`，该标签分布会诱发模型的类别偏置。因此新增了一轮候选视觉复核，并将数据状态与早期弱监督
实验区分记录。

候选复核的主要结果：

- 复核对象为 PS2.0 `training` 中 80 个独立来源组，每个来源组选择一张图片。
- 相对 teacher 标签，33 条风险等级和 77 条事件集合发生变化。
- 当前拆分为 48 条 LoRA train、16 条 validation、16 条独立 INT4 calibration。
- 三个开发 split 与冻结的 20 张 pilot 测试集均没有 `source_group_id` 交集。
- 当前标注来源为 `codex_visual_review_v1_single_pass`，属于候选标注，不是人工双人金标。

同时完成了 Jetson Transformers FP16 的独立 profiling。未插桩基线在 20 个测试样本上
全部完成，严格 JSON 有效率为 100%，端到端 p50/p90/p99 为 `9.38/14.12/27.94 s`，
风险等级准确率为 `35%`，事件 micro-F1 为 `0.341`。阶段插桩结果如下：

| 阶段 | p50 |
|---|---:|
| 图像预处理 | `27.60 ms` |
| 视觉编码 | `225.11 ms` |
| LLM prefill | `626.51 ms` |
| LLM decode | `17.04 s` |

profiling 在 hook 边界执行 CUDA 同步，因此插桩绝对时延不用于计算 runtime 加速比。
结果显示 decode 是主要耗时阶段，预处理不是当前优化重点。

### 2.7 2026-08-17：复核数据 LoRA 重训

服务器训练环境为 RTX 4090 D，使用 PyTorch `2.8.0+cu128`、Transformers `5.9.0`、
PEFT `0.18.0` 和 Accelerate `1.10.1`。基础模型仍固定为：

```text
Qwen/Qwen3-VL-2B-Instruct
revision: 89644892e4d85e24eaac8bacfd4f463576704203
```

为了缓解 teacher 标签全部为 `low` 的偏置，训练脚本增加了
`non_low_oversampling_factor`，仅对 train split 中非 `low` 样本进行复制，不修改
validation、calibration 或 test 数据。本轮 factor 为 2，48 个唯一训练样本扩展为 63
条有效训练记录。

三轮训练对照如下：

| 训练版本 | 唯一/有效 train | epoch / step | validation loss | 训练耗时 | 峰值 CUDA |
|---|---:|---:|---:|---:|---:|
| e1 | 48/48 | 1 / 12 | 1.2645 | 21.65 s | 5.249 GiB |
| e3 未平衡 | 48/48 | 3 / 36 | 0.8501 | 57.26 s | 5.273 GiB |
| e3、non-low x2 | 48/63 | 3 / 48 | 0.7220 | 80.84 s | 5.273 GiB |

冻结 20 样本的服务器结果：

| 模型状态 | 严格 JSON | 风险准确率 | 事件 micro-F1 | 现象 |
|---|---:|---:|---:|---|
| e1 adapter | 45% | 10% | 0 | 11 条输出在 256 token 处截断 |
| e3 未平衡 adapter | 100% | 35% | 0 | 20 条均预测为 `low` |
| e3、non-low x2 adapter | 100% | 50% | 0.182 | 恢复少量事件预测 |
| e3、non-low x2 merged | 100% | 45% | 0.100 | 与在线 adapter 结果不一致 |

本轮说明 validation loss 下降不能替代冻结测试集质量评测。过采样缓解了全 `low`
塌缩，但事件 F1 仍低于早期弱监督 LoRA 的 `0.389`。adapter 与 merged 的差异还需要
通过逐样本生成结果、权重合并过程和推理配置继续定位。

### 2.8 2026-08-17：领域 INT4 AWQ 重新量化与 Jetson 复测

量化环境使用 TensorRT Edge-LLM `v0.9.1` 对应工具链、ModelOpt `0.44.0`、
PyTorch `2.12.0+cu130`、Transformers `5.9.0` 和 datasets `4.8.5`。量化边界固定为：

- LLM backbone：W4A16 AWQ，group size 128；
- visual：FP16；
- `lm_head`：FP16；
- KV cache：不量化。

本轮使用 16 条无来源组泄漏的泊车领域文本作为 calibration，数据 SHA-256 为：

```text
0949bfb7649f74a0a537781e5e46363d9b76cb3b046ecf4b91b6cd02171f77f3
```

量化权重、ONNX 和 Jetson engine 均生成成功。Jetson 使用的 TensorRT Edge-LLM
固定 commit 为 `7f061f21f0a581ba234a1e233c9315b89d8e47d6`，新 LLM engine SHA-256 为：

```text
589d8ba247a93cdf794c86697bb5a5d5fe3387fee812744c51d09806912b3026
```

冻结 20 样本的板端结果：

| 指标 | 新领域 INT4 |
|---|---:|
| 后端完成 | 20/20 |
| 严格 JSON 有效率 | 4/20（20%） |
| 风险等级准确率 | 15% |
| 事件 micro-F1 | 0 |
| 端到端 p50 | 10.68 s |
| 聚合输出速率 | 7.32 token/s |
| RAM 峰值 | 5354 MB |
| GPU 利用率均值 | 81.88% |
| 输入功耗均值 | 9.26 W |
| GPU 峰值温度 | 62.97 °C |

20 条请求均获得后端响应，但其中 16 条输出被 Markdown `json` 代码围栏包裹，
不满足项目规定的严格 JSON 协议。评测器保留严格失败，不自动剥离代码围栏，以避免
掩盖量化后的格式遵循退化。

与旧的 128 条通用新闻文本校准 INT4 相比，新领域校准版本的性能基本相同，未带来
可确认的质量或性能收益。该实验被记录为“量化部署成功、质量验收失败”的负向结果，
不替换旧 INT4 作为当前性能对照。

### 2.9 当前状态与后续工作

截至 2026-08-17，项目已经完成以下端到端链路：

```text
固定模型 revision
  -> LoRA 训练 / 合并
  -> TensorRT Edge-LLM ONNX 导出
  -> Jetson engine 构建
  -> Edge-LLM HTTP 服务
  -> ParkSight Adapter
  -> StudyReport 与 tegrastats 证据
```

当前主要限制如下：

1. 20 张冻结测试集适合流程验收，样本规模不足以支撑稳定的领域质量结论。
2. 80 条复核标注仍是单轮候选标注，需要人工终审并补充更多六类风险事件。
3. 新 reviewed LoRA 的事件 micro-F1 低于旧弱监督 LoRA，不能描述为整体质量提升。
4. 新领域 INT4 的严格 JSON 有效率降至 20%，格式问题和任务质量问题仍未解决。
5. Jetson 8 GB 统一内存余量很小，FP16 engine 运行可能需要 headless、内存 compaction
   和临时 swap。

下一阶段应先完成候选数据人工终审与扩充，再在服务器上统一复验 Base、LoRA adapter
和 merged 模型；质量结果稳定后，再使用更大、更有代表性的泊车领域 calibration
重新执行 INT4，并将通过质量验收的版本部署到 Jetson。

详细命令、原始报告和证据索引见 `docs/record.md`、`docs/status.md` 和
`docs/progress.md`。

### 2.10 2026-08-29：复核 provenance、数据拆分与 Jetson 启动诊断

本轮将 80 条 Codex 单轮视觉复核候选标注重新固化为
`reports/label-review-20260829/ps80_codex_review_package_v1.jsonl`，并使用当前入口
完成候选数据生成验证。结果为 64 条 LoRA 数据（48 train、16 validation）和 16 条
INT4 calibration；LoRA 与 calibration 来源组交集为 0，开发数据与冻结测试集来源组
交集为 0。工作负载 identity 以及四份输入文件的 SHA-256 写入了
`ps80_candidate_dataset_summary.json`。

复核定稿入口新增以下约束：

- 候选和人工 `ParkingAssessment` 均必须通过严格 schema；
- `confirmed` 必须与候选完全一致，`corrected` 必须发生实际变化且填写 `review_note`；
- 定稿摘要输出风险等级准确率、事件 micro-precision/recall/F1、整体 assessment、
  风险等级、事件集合、证据和驾驶建议的修正统计；
- `prepare_reviewed_lora_dataset.py` 要求 CLI 显式提供 `--label-source`，避免人工金标
  被错误标记为 Codex 候选。

本轮无硬件测试增至 58 个并全部通过。Edge-LLM 服务入口新增
`--edge-llm-root` 和 `--plugin-path`，自动加入源码/pybind 路径并发现插件。

Jetson `192.168.137.187` 只读诊断结果：工作树为旧提交 `f362a43`，有 33 项未提交或
未跟踪改动，未执行覆盖或清理。临时补充 venv CUDA 库路径后，PyTorch `2.9.1`、CUDA
`12.6` 和 Transformers `4.57.6` 可导入。服务启动依次暴露了源码路径缺失、插件路径
缺失和图形桌面状态下 visual engine 申请约 `811 MiB` 连续内存失败三个问题；服务当前
未运行。尝试用 `ubuntu` 账号切换 headless 时因 sudo 需要密码被拒绝，root SSH 也未配置，
因此未继续修改系统状态。

当前下一步仍是人工终审 80 条候选标注；终审完成后使用 `human_confirmed_v1` 生成训练
和校准数据，再申请 Jetson headless/sudo 条件完成新的板端 smoke。

### 2.11 2026-08-29：正式训练的标注来源闸门

审计训练入口后发现，`configs/training/qwen3_vl_2b_lora_ps64_reviewed_v1.json` 原本只
固定了数据路径，没有在训练启动前检查记录的标注来源。该配置现在显式声明
`label_source=human_confirmed_v1`，并将 `allow_candidate_labels` 固定为 `false`。
`scripts/finetune_qwen3_vl_lora.py` 会检查配置来源、数据集是否单一来源且完全匹配，
并默认拒绝 `codex_visual_review_v1_single_pass`。因此当前 64 条候选训练数据会安全地
在训练前失败，而不会生成新的 LoRA adapter；这一步把“Codex 候选可用于开发验证”和
“人工终审后才可用于正式训练”明确分开。

本轮无硬件测试为 `61/61` 通过，包含来源匹配、混合来源拒绝、候选来源拦截和人工来源
接受四类训练入口测试。下一步是完成 80 条人工终审，将生成的数据路径和来源替换到
训练配置，再执行服务器 Base/LoRA/merged 对照评测。

同时增加 `scripts/inspect_review_package.py`，用于在人工复核过程中只读查看状态计数、
待处理 case_id、候选标签分布和最终化就绪状态；`--fail-on-incomplete` 可作为定稿前的
显式门禁，不会修改候选 package。

为支持先行验证开发链路，新增 `qwen3_vl_2b_lora_ps64_codex_candidate_v1` 训练配置和
服务器冻结集 study 配置，并将候选 LoRA 数据固定命名为
`data/processed/lora/ps64_codex_candidate_v1.jsonl`。候选训练必须使用独立产物目录和
`codex_candidate` 标识；该路径可以验证训练与 adapter 推理流程，但结果不进入正式质量
结论。

### 2.12 2026-08-30：Edge-LLM 静态部署预检

为减少 Jetson 启动失败的定位成本，`scripts/serve_edgellm.py` 新增 `--check-only`。
该模式只检查 LLM/visual engine、Edge-LLM checkout、pybind 和 plugin 路径，不导入或加载
GPU runtime。已在 Jetson 现有 `qwen3_vl_2b_fp16_i768_k1024` engine、
`/home/ubuntu/TensorRT-Edge-LLM/build/pybind` 和
`libNvInfer_edgellm_plugin.so` 上完成静态路径核对；实际服务仍因图形桌面统一内存条件
未重新启动；随后新增的真实 INT4 smoke 结果见 2.13。

同一入口现支持 LLM 与 visual engine 分目录传入，解决 LoRA/INT4 仅生成 LLM engine、
而视觉 engine 复用 FP16 版本时无法直接启动的问题。该模式已由无硬件测试覆盖；在
Jetson 上已核对 FP16、LoRA FP16、普通 INT4、领域 INT4 四种组合的 LLM/visual 目录，
四种预检均返回 `ready=true`；下一步仍需在 headless 条件下进行真实加载。

### 2.13 2026-08-30：领域 INT4 真实 HTTP smoke

在 Jetson 图形桌面保持运行的条件下，使用领域 INT4 LLM engine
`qwen3_vl_2b_int4_awq_ps16_v1_i768_k1024/llm`，复用 FP16 visual engine，显式传入
venv `site-packages`、Edge-LLM root、pybind 和 plugin 路径，并省略未启用的 weight
streaming 参数。实际加载日志确认 LLM engine、tokenizer、visual runner 和 CUDA graph
均初始化成功，Uvicorn 监听 `127.0.0.1:8000`，`/health` 返回 HTTP 200。

随后通过项目 `analyze_image` 入口发送 1 张 `ps2.0` 图片，请求返回 HTTP 200 和 91
output tokens。模型输出风险等级为 `low`、事件为 `vehicle_near_maneuver_path`，但原始
内容被 ```json 代码围栏包裹，严格 `ParkingAssessment` 解析因此记录为
`json_parse_error`。这证明领域 INT4 的部署和推理链路已经实际打通，同时保留了格式
遵循失败事实，没有把该请求写成业务成功。

原始日志已归档至 `reports/jetson-int4-smoke-20260830.log`，SHA-256 为
`9ee94b7b331368c7e7204288938eadd1bdd3f81e05e0c97209210bd7d77534b5`。临时服务已停止，
没有修改系统服务或远端仓库。后续应优先修正/重新验证严格 JSON 生成，再运行完整
`ps20_pilot` Study；FP16 图形桌面下的连续内存 OOM 仍独立存在。

### 2.14 2026-08-30：严格 JSON workload 修正与 Jetson A/B

为处理领域 INT4 的 Markdown JSON 代码围栏问题，新增
`configs/workloads/parking_risk_v2_strict_json.json`。该 workload 保留
`parking_risk_v1` schema、448x448 输入、生成参数和风险枚举，仅强化“只输出原始 JSON
对象”的边界约束，并通过缩短重复用户提示词控制 `i768` engine 的输入 token 预算。
其稳定 identity 为
`parking_risk_v2_strict_json@sha256:c4695a1bfa4d547f5ad90ec7697b82419dad12995829776e96c850707e15d1f4`。

在 Jetson 领域 INT4 LLM engine、复用 FP16 visual engine、冻结 `ps20_pilot_v1` 和同一
runtime 参数下完成完整 A/B。v1 运行结果为 20/20 后端完成、严格 JSON 有效率 20%、风险
等级准确率 15%、事件 micro-F1 0、端到端 p50 10.62 秒；v2 运行结果为 20/20、95%、35%、
0、p50 7.43 秒。v1 有 16 条 `json_parse_error`，v2 只有 1 条；v2 的平均输出长度也从
80.8 tokens 降至 57.8 tokens。v2 的格式有效率提升在完整冻结集上成立，但事件识别质量
没有改善，不能把该 workload 修正描述为领域能力提升。

第一次使用较长 v2 提示词时，服务明确报告输入 823 token 超出 engine 支持的 768 token；
压缩后请求通过，说明格式约束和输入预算需要共同设计。两次完整 study 均为单次重复，且
v2 先运行、v1 后运行，端到端 p50 仅作本轮描述性证据。

随后将 token 预算检查固化到 `scripts/inspect_prompt_contract.py`。使用 Jetson 实际
Qwen3-VL processor、同一图片和 engine chat template，v1/v2 输入 token 数为 `735/700`；
以 `--max-input-tokens 768` 运行时两者均通过门禁，且 `message_contract.messages_equal`
均为 `true`。报告分别为
`reports/prompt-contract/qwen3_vl_v1_i768_budget_20260830.json`（SHA-256
`d7d11b4192f97d9d8035bfc53d911299a673c0c8087af996e8a2547fe5ffc0cf`）和
`reports/prompt-contract/qwen3_vl_v2_i768_budget_20260830.json`（SHA-256
`247f66553562cb5d7de3aae4e85ba07bd19ade626c002540ddacc26fa967a663`）。

新 study 配置为
`configs/studies/jetson_edgellm_int4_awq_ps16_v1_ps20_pilot_strict_json.json`，原始日志
为 `reports/jetson-int4-ps20-strict-json-20260830.log`；StudyReport 为
`reports/jetson_edgellm_int4_awq_ps16_v1_ps20_pilot_strict_json_i768_k1024.json`，SHA-256
为 `9f75374756305d820baf8efd635a5ef709dc867a453e2632f426c78d897c1cc0`。下一步应在人工
确认数据完成后重新训练/量化，并继续用完整冻结集比较 JSON 有效率、风险准确率、事件
micro-F1、输出 token 数和端到端分位数。

### 2.15 2026-08-30：80 条 Codex 候选样本开发评测

为验证候选数据能否跑通完整识别流程，在同一 Jetson INT4 engine、同一 v2 严格 JSON
workload 和同一运行时参数下，分别评测 `ps80_development_v1` 的 64 条 train 与 16 条
validation。参考 annotation 是 Codex 候选结果，不是人工终审金标，因此该实验仅用于
流程验证和错误分析。

合并 80 条结果后，后端完成率为 `80/80`，严格 JSON 有效率为 `77/80=96.25%`，候选
标签上的风险等级准确率为 `47/80=58.75%`，事件 micro-F1 为 `0`，不安全建议率为
`23/80=28.75%`，3 条失败均为 `json_parse_error`，且均位于 train 分片。train 分片
自身为严格 JSON `95.31%`、风险准确率 `57.81%`；validation 分片为 `100%` 和 `62.50%`。
结果表明当前链路可以稳定处理 80 条输入，但模型对风险事件的输出仍未达到可用水平。

新增可复现配置为
`configs/studies/jetson_edgellm_int4_awq_ps16_ps80_codex_candidate_train_strict_json.json`
和 validation 版本。报告分别为
`reports/jetson_edgellm_int4_awq_ps16_ps80_codex_candidate_train_strict_json_i768_k1024.json`
（SHA-256 `21a3ef7dd7f0a4d76f0846a03448cca6c296b2458c0a1ec065b70a5958d10671`）和
`reports/jetson_edgellm_int4_awq_ps16_ps80_codex_candidate_validation_strict_json_i768_k1024.json`
（SHA-256 `8edf7cd695cfffd919d521653bd7877421d470396705a12e089e76fb11ebb4dd`）。

### 2.16 2026-08-30：候选 LoRA 训练、合并与服务器对照

在本地 RTX 4060 上创建独立 `.venv-train`，安装 PyTorch `2.8.0+cu128`、Transformers
`5.9.0`、PEFT `0.18.0` 和 Accelerate `1.10.1`。从 Jetson 缓存复制的 Qwen3-VL
权重大小为 `4,255,140,312` 字节，SHA-256 为
`7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`，与板端缓存一致。
processor 和 BF16 模型加载成功，CUDA 可用且 BF16 支持。

使用 `configs/training/qwen3_vl_2b_lora_ps64_codex_candidate_v1.json` 完成候选训练：
48 个唯一训练样本，non-low 过采样后 63 条，3 epochs、48 个 optimizer steps，validation
loss 为 `0.723180890083313`，峰值 CUDA 显存 `5.272403240203857 GiB`，耗时
`145.891 s`。adapter 输出位于被忽略的 `artifacts/` 目录。

在固定 `ps20_pilot_v1`、`parking_risk_v1` 和同一 model revision 上，base 对照为风险
准确率 `35%`、事件 micro-F1 `0.350`；candidate adapter 为 `50%`、`0.1818`，两者
严格 JSON 均为 `100%`。candidate 提高了风险等级命中，但事件错误增加，不能作为正式
模型改进。随后使用
`configs/flows/merge_qwen3_vl_2b_lora_ps64_codex_candidate_v1.json` 完成合并，并以
`configs/studies/server_transformers_merged_lora_ps64_codex_candidate_v1_ps20_pilot.json`
复测；merged 与 adapter 的 20 条 case 顺序、原始输出和质量指标完全一致。

### 2.17 2026-08-30：候选结果人工复核清单

为减少人工终审的整理成本，新增 `scripts/build_candidate_error_review.py`，合并 80 条
Jetson train/validation StudyReport，并与 `ps80_reviewed_v1` 候选 annotation 逐 case
对齐。输出清单包含图片引用、来源组、候选 assessment、模型 assessment、原始输出、失败
原因、风险等级是否匹配、事件 false positive/false negative 以及复核优先级。

当前清单覆盖 80 个 case，77 条 JSON 有效，47 条风险等级匹配，47 条事件完全匹配，30
个 case 存在事件差异，33 个 case 被标为高优先级；事件差异的 false positive 为 0，
漏检主要集中在 `vehicle_near_maneuver_path`（20）、`narrow_passage`（18）和
`visibility_occlusion`（9）。清单输出为
`reports/label-review-20260830/ps80_candidate_error_review_v1.json`，仅支持人工复核，
不会改变候选 annotation 或自动生成 `human_confirmed_v1`。

### 2.18 2026-08-30：候选 LoRA 的 80 条服务器开发集评测

为确认“候选标注—LoRA 训练—服务器评测”闭环，复用本地 RTX 4060、同一 Qwen3-VL
revision、同一 `parking_risk_v2_strict_json` workload 和已生成的候选 adapter，分别对
`ps80_development_v1` 的 64 条 train、16 条 validation 执行一次评测。参考标签仍为
`ps80_reviewed_v1` 的 Codex 候选结果，因而本轮属于开发验证，不是人工金标质量验收。

| 分片 | 样本 | 严格 JSON | 风险准确率 | 事件 micro-F1 | 不安全建议率 | 端到端 p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 64 | 100% | 57.81% | 0 | 32.81% | 3424 ms |
| validation | 16 | 100% | 62.50% | 0 | 18.75% | 4087 ms |
| 合计 | 80 | 100% | 58.75% | 0 | 30.00% | 分片口径 |

两处分片均完成严格 JSON 解析，且 validation 风险准确率没有低于 train；但两者事件
micro-F1 都为 0，不能据此判断模型已学会六类风险事件。相较冻结 `ps20_pilot_v1` 的
候选 adapter 结果，本轮 workload 与参考标签不同，不能直接进行质量横向比较；正式
结论仍必须以人工确认的 `human_confirmed_v1` 和冻结测试集为准。

本轮配置为
`configs/studies/server_transformers_lora_ps64_codex_candidate_v1_ps80_train_strict_json.json`
和 validation 版本。报告保存在本地忽略目录 `reports/`，SHA-256 分别为
`d3bea513b671dfd5d84f034be1d5d1ec9b0f4bd259bcd7279b843cb067c853bf` 和
`d1c81a98dfba0ed0f9b9ef3234988627aab6a8cd76ebc7052fc16c9b51afae87`。
