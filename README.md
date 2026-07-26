# Jetson AutoRisk-VLM

面向车端低速行车场景的单图风险理解、领域 LoRA 适配与 Jetson 运行时评测项目。

```text
RoadCase -> RiskRuntime -> InferenceRecord -> StudyReport
```

## 项目结构

```text
src/auto_risk_vlm/
  assessment/     # 严格风险 JSON
  casebook/       # 道路样本、标注和数据划分
  inference/      # Transformers / TensorRT Edge-LLM Adapter
  studies/        # 质量、性能、环境和失败归因
  app/            # 命令入口
configs/
  workloads/      # 冻结 prompt 与生成参数
  studies/        # 实验配置
data/
  manifests/      # 元数据
  annotations/    # 人工标注
  raw/            # 本地图片
artifacts/        # 模型和 engine 生成物
reports/          # StudyReport
scripts/          # 显式流程入口
tests/            # 无硬件单元测试
docs/              # 项目文档
```

## 主要 Module

- `assessment`：将模型 JSON 解析为 `RiskAssessment`。
- `casebook`：将图片、标注、来源组与数据集划分组织为 `RoadCase`。
- `inference`：通过 `RiskRuntime.analyze(case, workload)` 生成 `InferenceRecord`。
- `studies`：通过 `StudyRunner.run(casebook, runtime, study)` 生成 `StudyReport`。

完整架构见 [docs/architecture.md](docs/architecture.md)，领域术语见 [CONTEXT.md](CONTEXT.md)。
