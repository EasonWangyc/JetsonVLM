# ParkSight-VLM

面向低速泊车场景的单图风险理解、领域 LoRA 适配与 Jetson 运行时评测项目。

```text
ParkingCase -> RiskRuntime -> InferenceRecord -> StudyReport
```

## 场景与输出

输入一张泊车场景图片，输出包含风险等级、事件、证据和驾驶建议的严格 JSON。事件集合覆盖行人接近行驶路径、车辆接近行驶路径、固定障碍物、狭窄通道、可见性遮挡和车位冲突。

## 项目结构

```text
src/parksight_vlm/
  assessment/     # 严格泊车风险 JSON
  casebook/       # 泊车样本、标注和数据划分
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

- `assessment`：将模型 JSON 解析为 `ParkingAssessment`。
- `casebook`：将图片、标注、来源组与数据集划分组织为 `ParkingCase`。
- `inference`：通过 `RiskRuntime.analyze(case, workload)` 生成 `InferenceRecord`。
- `studies`：通过 `StudyRunner.run(casebook, runtime, study)` 生成 `StudyReport`。

完整架构见 [docs/architecture.md](docs/architecture.md)，领域术语见 [CONTEXT.md](CONTEXT.md)。
