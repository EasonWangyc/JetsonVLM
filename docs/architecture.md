# 系统架构

## 架构中心

系统围绕泊车样本、泊车风险评估、推理记录和研究报告组织。模型运行时、量化精度、adapter 和板端环境通过推理记录与研究报告进入同一条数据流。

```text
ParkingCaseCatalog -> ParkingCase -> RiskRuntime -> InferenceRecord -> StudyRunner -> StudyReport
```

## 深 Module

| Module | Interface | 输入 | 输出 | 内部职责 |
| --- | --- | --- | --- | --- |
| `assessment` | `ParkingAssessment.from_mapping(payload)` | JSON mapping | `ParkingAssessment` | 字段完整性、枚举、文本、事件集合与 schema version 校验 |
| `casebook` | `ParkingCaseCatalog.load()` / `ParkingCaseCatalog.validate()` | manifest、标注、工作负载选择 | `ParkingCase` 集合 | 来源组约束、数据集划分、冻结测试集与图片引用解析 |
| `inference` | `RiskRuntime.analyze(case, workload)` | `ParkingCase`、冻结工作负载 | `InferenceRecord` | 输入准备、Adapter 调用、输出解析、阶段计时与运行失败记录 |
| `studies` | `StudyRunner.run(casebook, runtime, study)` | casebook、runtime、研究配置 | `StudyReport` | 推理记录归档、任务质量、性能分位数、环境快照与失败归因 |

每个 Module 通过一个小 Interface 向调用方提供深度。`inference` 内部包含 `TransformersRuntime` 与 `EdgeLlmRuntime` 两个 Adapter。`studies` 通过 `RiskRuntime` Interface 运行两类 Adapter 的任务与性能研究。

## 对象关系

```text
ParkingCase
  - case_id
  - image_ref
  - source_group_id
  - split
  - reference_assessment

ParkingAssessment
  - schema_version
  - risk_level
  - events
  - evidence
  - driver_advice

InferenceRecord
  - case_id
  - runtime_identity
  - workload_identity
  - assessment | failure
  - stage_timings
  - resource_snapshot

StudyReport
  - study_identity
  - environment_snapshot
  - quality_metrics
  - performance_metrics
  - failure_summary
```

`ParkingAssessment` 作为人工标注和模型解析结果的共享结构。`InferenceRecord` 保存单次执行事实。`StudyReport` 聚合多个推理记录，保留实验配置和环境快照。

## 目录职责

```text
src/parksight_vlm/
  assessment/
    model.py          # ParkingAssessment、枚举和校验错误
  casebook/
    model.py          # ParkingCase、来源组和数据集划分
    catalog.py        # ParkingCaseCatalog Interface 与 manifest 读取
  inference/
    runtime.py        # RiskRuntime Interface、InferenceRecord 与失败类型
    transformers.py   # TransformersRuntime Adapter
    edge_llm.py       # EdgeLlmRuntime Adapter
  studies/
    model.py          # Study 与 StudyReport
    runner.py         # StudyRunner
    quality.py        # 质量指标
    performance.py    # 时延、资源和环境汇总
  app/
    analyze_image.py  # 单图分析命令
    run_study.py      # 批量研究命令
configs/
  workloads/parking_risk_v1.json
  studies/transformers_base.json
  studies/edgellm_fp16.json
data/
  manifests/parking_risk_v1.jsonl
  annotations/parking_risk_v1.jsonl
  raw/
artifacts/
reports/
scripts/
tests/
  assessment/
  casebook/
  inference/
  studies/
```

## 固定工作负载

`workload` 固定任务语义、prompt、生成参数、输入尺寸、schema version 和泊车风险事件集合。`study` 记录 runtime、模型 revision、adapter revision、精度、样本选择、运行次数和功耗模式。

泊车风险事件集合：

- `vru_near_maneuver_path`
- `vehicle_near_maneuver_path`
- `fixed_obstacle_near_path`
- `narrow_passage`
- `visibility_occlusion`
- `parking_space_conflict`

建议 `driver_advice` 使用：`maintain_observation`、`slow_down`、`yield`、`prepare_to_stop`、`change_maneuver_when_safe`。

`workload` 保持任务可比性；`study` 表达一次实验的运行条件。该拆分使 FP16、LoRA 和 INT4 结果在同一工作负载下可直接比较。

## 测试 seam

1. `ParkingAssessment.from_mapping`：结构化内容的解析与校验。
2. `ParkingCaseCatalog.validate`：来源组、样本标识与数据集划分的校验。
3. `RiskRuntime.analyze`：Adapter 对泊车样本和工作负载的实际执行。
4. `StudyRunner.run`：推理记录到研究报告的聚合。

首批实现围绕前两个 seam 展开。运行时 Adapter 与研究运行器接入后，后两个 seam 形成显式集成测试。
