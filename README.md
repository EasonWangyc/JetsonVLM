# JetsonVLM

<p>
  <img src="https://img.shields.io/badge/Built%20with-Codex-412991" alt="Built with Codex">
<p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/TypeScript-5.9-3178C6" alt="TypeScript 5.9">
  <img src="https://img.shields.io/badge/Qwen3--VL-2B--Instruct-6E56CF" alt="Qwen3-VL 2B Instruct">
  <img src="https://img.shields.io/badge/TensorRT--Edge--LLM-v0.9.1-76B900" alt="TensorRT Edge-LLM v0.9.1">
  <img src="https://img.shields.io/badge/JetPack-6.2.2-76B900" alt="JetPack 6.2.2">
</p>

面向低速泊车场景的视觉语言模型研究与部署项目。项目围绕
`Qwen/Qwen3-VL-2B-Instruct` 搭建风险理解、领域 LoRA 适配、INT4 AWQ 量化、
TensorRT Edge-LLM 部署和可审计评测链路。

## 技术栈与固定版本

| 组件 | 版本或固定身份 | 用途 |
|---|---|---|
| Python | `>=3.10` | 项目运行环境 |
| TypeScript | `5.9` | 项目协作/工具链标识；核心运行代码为 Python |
| Qwen3-VL | `Qwen/Qwen3-VL-2B-Instruct`，revision `89644892e4d85e24eaac8bacfd4f463576704203` | 基础视觉语言模型 |
| Transformers | 服务器 `5.9.0`；Jetson `4.57.6` | 训练、质量参考和 Jetson FP16 基线 |
| PyTorch | 服务器 `2.8.0+cu128`；Jetson `2.9.1` | 训练与推理 |
| PEFT / Accelerate | `0.18.0` / `1.10.1` | LoRA 训练 |
| ModelOpt | `0.44.0` | INT4 AWQ 量化 |
| TensorRT Edge-LLM | `v0.9.1`，commit `7f061f21f0a581ba234a1e233c9315b89d8e47d6` | Jetson runtime、导出和 engine 构建 |
| JetPack / CUDA / TensorRT | `6.2.2` / `12.6` / `10.3` | Jetson Orin Nano 板端环境 |


默认评测 workload 为 `configs/workloads/parking_risk_v2_strict_json.json`，其
workload identity 为：

```text
parking_risk_v2_strict_json@sha256:6ca953643f38a13b579a77090c77d3fca30d3ba9a1b181d88ae11692ea150fec
```

该 workload 保留 `parking_risk_v1` schema，输入尺寸为 `448x448`，
`max_new_tokens=256`，`do_sample=false`，并要求模型直接输出原始 JSON。

## 功能与输出

输入为单张泊车场景图片，输出为一个严格校验的 `ParkingAssessment`：

```json
{
  "schema_version": "parking_risk_v1",
  "risk_level": "medium",
  "events": ["narrow_passage"],
  "evidence": ["车辆两侧空间较窄，可用横向通行区域有限。"],
  "driver_advice": ["slow_down", "maintain_observation"]
}
```

字段约束如下：

- `risk_level`：只能为 `low`、`medium` 或 `high`，表示场景整体风险。
- `events`：固定六类事件的多标签数组，可以为空。
- `evidence`：与图片可见线索对应的简短说明，不是思维链。
- `driver_advice`：枚举式安全提示，不是车辆控制指令。
- `prepare_to_stop`：表示高风险场景下随时准备停车，不表示“准备泊入车位”。

核心调用链为：

```text
ParkingCase -> RiskRuntime -> InferenceRecord -> StudyReport
```

支持服务器 Transformers 正确性参考、Jetson Transformers FP16、TensorRT Edge-LLM
FP16、领域 LoRA、合并模型和 LLM backbone INT4 AWQ 研究。服务器结果用于质量和误差
分析；板端性能只比较 Jetson 上的 runtime。

## 最终已核验结果

### Jetson runtime 对比

结果来自冻结 `ps20_pilot_v1` 测试集，20 个样本、单次重复。`p50` 为端到端时延；
服务器 GPU 时延不用于推导 Jetson 加速比。

| Runtime / 模型 | 后端完成 | 严格 JSON | 风险准确率 | 事件 micro-F1 | 端到端 p50 | 聚合输出速率 |
|---|---:|---:|---:|---:|---:|---:|
| Jetson Transformers FP16 | 20/20 | 100% | 35% | 0.341 | 9.38 s | 未报告 |
| Edge-LLM Base FP16 | 20/20 | 100% | 35% | 0.359 | 50.75 s | 1.48 token/s |
| Edge-LLM 旧 LoRA FP16 | 20/20 | 100% | 35% | 0.389 | 30.60 s | 1.44 token/s |
| Edge-LLM 旧通用校准 INT4 | 20/20 | 100% | 35% | 0 | 10.52 s | 7.33 token/s |
| Edge-LLM 新领域校准 INT4 | 20/20 | 20% | 15% | 0 | 10.68 s | 7.32 token/s |

旧 INT4 相对 Edge-LLM Base FP16 的端到端延迟约降低至五分之一，engine 体积减少约
60.5%；但量化后的任务质量仍需单独验收。旧 LoRA 的较低 p50 同时伴随约 41.8% 的
输出 token 减少，不能直接解释为 runtime 加速。



## 项目结构

```text
src/parksight_vlm/
  assessment/       # ParkingAssessment、严格 JSON schema 和语义审计
  casebook/         # ParkingCase、annotation、manifest 和 split 校验
  inference/        # Transformers / TensorRT Edge-LLM adapters
  studies/          # InferenceRecord、指标、性能和 StudyReport
  app/              # 单图分析、Study 入口和 runtime factory
configs/
  workloads/        # 冻结 prompt、生成参数、输入尺寸和 schema
  studies/          # 质量/性能 Study 配置
  flows/             # 训练、合并、导出、量化和 engine 构建配置
data/
  manifests/        # 可提交的样本元数据
  annotations/      # 可提交的标注
  raw/               # 本地图片输入，默认不进入 Git
scripts/             # 训练、评测、导出、量化、构建和服务脚本
tests/               # 无硬件单元测试
docs/                # 架构、数据、评测、部署和过程记录
artifacts/           # adapter、ONNX、权重和 engine，本地生成
reports/             # StudyReport 和运行证据，本地生成
models/              # 本地模型权重，默认不进入 Git
```

核心接口：

- `ParkingAssessment.from_mapping(payload)`：解析、校验和序列化严格 JSON。
- `ParkingCaseCatalog.load()` / `validate()`：加载样本并校验来源组、split 和图片引用。
- `RiskRuntime.analyze(case, workload)`：执行一次推理并保留成功或失败事实。
- `StudyRunner.run(casebook, runtime, study)`：聚合推理记录并生成 `StudyReport`。

## 使用方法

### 安装与无硬件测试

在已准备好项目依赖的环境中：

```powershell
py -3.12 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -e .
$env:PYTHONPATH = "src"
& ".\.venv\Scripts\python.exe" -m unittest discover -s tests -v
```

### 单图推理

Transformers 运行时示例：

```powershell
$env:PYTHONPATH = "src"
& ".\.venv\Scripts\python.exe" -m parksight_vlm.app.analyze_image `
  --image data\raw\example.jpg `
  --workload configs\workloads\parking_risk_v2_strict_json.json `
  --runtime transformers `
  --backend-revision transformers-5.9.0 `
  --model-revision 89644892e4d85e24eaac8bacfd4f463576704203 `
  --precision bf16
```

使用已训练 LoRA 时增加 `--adapter-path <adapter-directory>`；使用 Jetson Edge-LLM
HTTP runtime 时将 `--runtime` 改为 `tensorrt_edge_llm_http`，并提供 `--edge-url`。

### 运行配置化 Study

```powershell
$env:PYTHONPATH = "src"
& ".\.venv\Scripts\python.exe" -m parksight_vlm.app.run_study `
  --config configs\studies\server_transformers_base_ps20_pilot.json
```

Study 会将完整报告写入配置指定的 `reports/` 路径，包含 JSON、风险、事件、时延、
内存和失败归因。Jetson 研究使用 `configs/studies/` 中对应的 Edge-LLM 配置，并在
板端执行。

### 训练前门禁

正式 LoRA 训练前可先只做数据、来源、split、图片、workload 和模型路径校验：

```powershell
$env:PYTHONPATH = "src"
& ".\.venv-train\Scripts\python.exe" scripts\finetune_qwen3_vl_lora.py `
  --config configs\training\qwen3_vl_2b_lora_ps64_reviewed_v1.json `
  --validate-only
```

训练、合并、导出、量化和 engine 构建均通过 `scripts/` 与 `configs/flows/` 的显式
配置执行；完整操作顺序见 [`docs/operations.md`](docs/operations.md)，Jetson 部署细节
见 [`docs/edgellm-deployment.md`](docs/edgellm-deployment.md)。

## 文档入口

- [当前实现状态](docs/status.md)
- [项目进展](docs/progress.md)
- [完整执行记录](docs/execution-report.md)
- [系统架构](docs/architecture.md)
- [数据说明](docs/data.md)
- [评测口径](docs/evaluation.md)
- [操作入口](docs/operations.md)
- [Edge-LLM 部署](docs/edgellm-deployment.md)
- [项目记录](docs/personal_record.md)
