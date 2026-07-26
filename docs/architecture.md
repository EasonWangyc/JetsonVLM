# 系统架构

## 架构中心

系统围绕四类事实组织：道路样本、风险评估、推理记录和研究报告。模型运行时、量化精度、adapter 和板端环境都通过推理记录与研究报告进入同一条数据流。

```text
CaseCatalog -> RoadCase -> RiskRuntime -> InferenceRecord -> StudyRunner -> StudyReport
```

## 深 Module

| Module | Interface | 输入 | 输出 | 内部职责 |
| --- | --- | --- | --- | --- |
| `assessment` | `RiskAssessment.from_mapping(payload)` | JSON mapping | `RiskAssessment` | 字段完整性、枚举、文本、事件集合与 schema version 校验 |
| `casebook` | `CaseCatalog.load()` / `CaseCatalog.validate()` | manifest、标注、工作负载选择 | `RoadCase` 集合 | 来源组约束、数据集划分、冻结测试集与图片引用解析 |
| `inference` | `RiskRuntime.analyze(case, workload)` | `RoadCase`、冻结工作负载 | `InferenceRecord` | 输入准备、Adapter 调用、输出解析、阶段计时与运行失败记录 |
| `studies` | `StudyRunner.run(casebook, runtime, study)` | casebook、runtime、研究配置 | `StudyReport` | 预测记录归档、任务质量、性能分位数、环境快照与失败归因 |

每个 Module 通过一个小 Interface 向调用方提供深度。`inference` 内部存在两个 Adapter：`TransformersRuntime` 与 `EdgeLlmRuntime`。`studies` 只依赖 `RiskRuntime` Interface，因此两个 Adapter 使用同一套任务与性能研究流程。

## 对象关系

```text
RoadCase
  - case_id
  - image_ref
  - source_group_id
  - split
  - reference_assessment

RiskAssessment
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

`RiskAssessment` 作为人工标注和模型解析结果的共享结构。`InferenceRecord` 保存单次执行事实。`StudyReport` 聚合多个推理记录，保留实验配置和环境快照。

## 目录职责

```text
src/auto_risk_vlm/
  assessment/
    model.py          # RiskAssessment、枚举和校验错误
  casebook/
    model.py          # RoadCase、来源组和数据集划分
    catalog.py        # CaseCatalog Interface 与 manifest 读取
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
  workloads/road_risk_v1.json
  studies/transformers_base.json
  studies/edgellm_fp16.json
data/
  manifests/road_risk_v1.jsonl
  annotations/road_risk_v1.jsonl
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

## 工作负载与研究配置

`workload` 固定任务语义、prompt、生成参数、输入尺寸、schema version 和事件集合。`study` 记录 runtime、模型 revision、adapter revision、精度、样本选择、运行次数和功耗模式。

`workload` 保持任务可比性；`study` 表达一次实验的运行条件。该拆分使 FP16、LoRA 和 INT4 结果在同一工作负载下可直接比较。

## 测试 seam

1. `RiskAssessment.from_mapping`：结构化内容的解析与校验。
2. `CaseCatalog.validate`：来源组、样本标识与数据集划分的校验。
3. `RiskRuntime.analyze`：Adapter 对道路样本和工作负载的实际执行。
4. `StudyRunner.run`：推理记录到研究报告的聚合。

前三个阶段的实现从前两个 seam 开始。运行时 Adapter 出现后，第三和第四个 seam 形成显式集成测试。
