# Jetson AutoRisk-VLM 仓库指南

## 项目范围

本仓库实现车端低速行车场景的单图风险理解、领域 LoRA 适配与 Jetson 部署评测。业务主链路为：

```text
RoadCase -> RiskRuntime -> InferenceRecord -> StudyReport
```

目标模型为 `Qwen/Qwen3-VL-2B-Instruct`。Transformers 提供正确性参考，TensorRT Edge-LLM 提供 Jetson 运行时。项目以 FP16 基线、领域 LoRA、合并模型和 LLM backbone INT4 为实验序列，并记录每个阶段的实际证据。

项目产出驾驶安全提示。风险事件集合包括：

- `vru_near_driving_path`
- `vehicle_cut_in_or_stop`
- `lane_obstruction`
- `road_work_zone`
- `intersection_occlusion`
- `low_visibility`

文档和项目描述围绕已实现功能、输入输出、技术路径、评测口径和实测证据展开，采用项目中心的客观表述。

## 领域对象与 Module

- `RoadCase`：一张道路图片、来源组、数据集划分和人工标注的组合。
- `RiskAssessment`：风险等级、风险事件、证据和驾驶建议构成的严格 JSON 内容。
- `InferenceRecord`：一次运行时执行的输入标识、配置、`RiskAssessment`、时延与失败事实。
- `StudyReport`：一个冻结工作负载上的质量、性能、环境与失败归因汇总。

核心 Module 及其 Interface：

| Module | Interface | 职责 |
| --- | --- | --- |
| `assessment` | `RiskAssessment.from_mapping(payload)` | 解析、校验和序列化严格 JSON |
| `casebook` | `CaseCatalog.load()` / `CaseCatalog.validate()` | 管理道路样本、标注、来源组和数据集划分 |
| `inference` | `RiskRuntime.analyze(case, workload) -> InferenceRecord` | 适配 Transformers 与 TensorRT Edge-LLM，保留运行时事实 |
| `studies` | `StudyRunner.run(casebook, runtime, study) -> StudyReport` | 统一完成质量评测、性能采集和报告生成 |

`app` 只负责组合 Module 并提供命令入口。`scripts` 只承担训练、合并、导出和构建等显式流程。

## 目录结构

```text
src/auto_risk_vlm/
  assessment/       # RiskAssessment 与 JSON schema
  casebook/         # RoadCase、标注、manifest 与划分校验
  inference/        # RiskRuntime Interface 和具体 Adapter
  studies/          # InferenceRecord、评测、benchmark 与环境快照
  app/              # 单图分析与实验运行入口
configs/
  workloads/        # 冻结 prompt、生成参数和 schema 版本
  studies/          # 基线、LoRA、FP16、INT4 实验配置
data/
  manifests/        # 可提交的样本元数据
  annotations/      # 可提交的人工标注
  raw/              # 本地图片输入
artifacts/          # adapter、导出模型和 engine
reports/            # StudyReport 与图表
scripts/            # train、merge、export、build 等显式入口
tests/              # 围绕 Module Interface 的无硬件测试
docs/               # 架构、数据卡、部署和报告说明
```

## 实施顺序

1. 实现 `RiskAssessment`、`RoadCase`、`CaseCatalog` 和冻结工作负载配置。
2. 建立 Transformers Adapter，生成 `InferenceRecord` 并运行质量研究。
3. 建立 TensorRT Edge-LLM FP16 Adapter，完成 Jetson 运行时研究。
4. 基于冻结测试集的领域误差分析训练 LoRA，完成合并模型的同口径研究。
5. 生成 INT4 LLM backbone 版本，完成质量、时延、内存、功耗与温度对比。

每个阶段保存模型 revision、adapter revision、精度、输入尺寸、prompt、生成参数、数据集版本、执行命令、环境快照和结果状态。

## 数据与评测

- `source_group_id` 将同一视频、连续采集序列或不可分数据来源的样本归入同一数据集划分。
- 冻结测试集在 LoRA 调参、量化校准和阈值调整之前确定，并保留人工标注、数据版本和来源信息。
- 失败记录使用明确类别，包括 JSON 解析失败、模型拒答、超时、显存不足、算子支持状态和任务判断错误。
- `StudyReport` 记录 JSON 有效率、风险等级准确率、事件 micro-F1、不安全建议率、按类别错误、冷启动、预处理/视觉编码/prefill/decode/端到端时延、p50/p90/p99、首 token 延迟、tokens/s、峰值内存、功耗、温度和失败样例。

## 编码、测试与操作

- 使用 Python 3.10+、四空格缩进、类型标注、`pathlib.Path`、dataclass 和 `snake_case`。
- 单元测试围绕 Module 的 Interface 编写，并可在无模型权重、CUDA、摄像头和 Jetson 的环境中运行。
- 模型与板端验证属于显式集成测试，运行记录包含 backend 状态和失败原因。
- 标准无硬件测试使用 `uv run python -m unittest discover -s tests`；修改后运行 `git diff --check`。
- 项目配置固定依赖和模型版本。ONNX Runtime 用于导出和数值一致性研究，TensorRT Edge-LLM 用于 Jetson 运行时研究。
- 系统依赖、远程 Jetson、大模型权重、Git 提交、推送和历史操作遵循明确授权流程。
