# JetsonVLM

基于 Jetson Orin Nano 与 TensorRT Edge-LLM 的 Qwen3-VL-2B 泊车风险理解、LoRA
领域适配、INT4 AWQ 量化部署与评测项目。

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
docs/             # 项目文档
```

## 主要 Module

- `assessment`：将模型 JSON 解析为 `ParkingAssessment`。
- `casebook`：将图片、标注、来源组与数据集划分组织为 `ParkingCase`。
- `inference`：通过 `RiskRuntime.analyze(case, workload)` 生成 `InferenceRecord`。
- `studies`：通过 `StudyRunner.run(casebook, runtime, study)` 生成 `StudyReport`。

## 实验基线

项目区分三种实验角色：

- 服务器 Transformers：正确性参考、完整质量研究、误差分析和 LoRA 训练。
- Jetson Transformers FP16：板端原生框架基线，记录可运行性、OOM、时延和内存。
- Jetson TensorRT Edge-LLM：最终部署 runtime，与 Jetson Transformers 在同一工作负载下比较 FP16 和后续 INT4 结果。

服务器性能不用于证明 Jetson 加速收益；部署性能结论来自 Jetson 同机实验。

## 当前实测结论

固定 revision 的 Qwen3-VL-2B 已完成服务器 LoRA 训练、adapter 合并和 TensorRT
Edge-LLM ONNX 导出，以及 Jetson 上 Base FP16、LoRA 合并模型和 INT4 engine 构建与
冻结 20 样本评测。Jetson Base FP16/LoRA 的严格 JSON 有效率均为 100%，事件
micro-F1 从 0.359 提升至 0.389；INT4 相比 Base FP16 的平均端到端延迟由 53.64 秒
降至 10.69 秒、engine 体积降低 60.5%，但通用文本 AWQ 校准导致事件 micro-F1
退化为 0。项目将格式有效性、任务质量和部署性能分别记录，避免把 runtime 成功等同于
业务质量达标。详见 [docs/status.md](docs/status.md)。

## 开始使用

无硬件验证：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

项目不会在导入或测试时下载模型、启动 GPU 任务或构建 engine。模型操作通过显式命令
进入，并保留输入、输出、日志和结果状态。

完整架构见 [docs/architecture.md](docs/architecture.md)

操作入口见[docs/operations.md](docs/operations.md)

当前实现与待实测边界见[docs/status.md](docs/status.md)

评测口径见[docs/evaluation.md](docs/evaluation.md)

领域术语见 [CONTEXT.md](CONTEXT.md)，

数据 JSONL 约定见 [docs/data.md](docs/data.md)

当前环境、模型缓存、执行命令和下一阶段见 [docs/progress.md](docs/progress.md)

从项目开始至今的命令、结果与证据见 [docs/execution-report.md](docs/execution-report.md)。
