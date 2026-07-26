# 评测口径

## 冻结单位

一次可比较研究由以下身份共同确定：

- workload SHA-256；
- 数据集 manifest、annotation 与 split；
- backend 名称及 revision；
- model id 及不可变 revision；
- LoRA adapter revision；
- 精度、power mode 与 repetitions。

任一项变化都应产生新的 `study_id` 和报告，不覆盖既有证据。

## 实验角色与可比性

| 实验 | 主要结论 | 不用于 |
| --- | --- | --- |
| 服务器 Transformers | 模型、prompt、严格 JSON 和任务质量的正确性参考 | Jetson 部署性能结论 |
| Jetson Transformers FP16 | 原生框架在目标板上的可运行性、OOM、质量和性能基线 | 服务器训练吞吐比较 |
| Jetson TensorRT Edge-LLM FP16/INT4 | 最终部署质量与板端性能 | 替代服务器误差分析 |

质量一致性可以在三个 runtime 间比较，但必须固定模型 revision、workload 和数据集。
时延、内存、功耗和温度的加速结论只比较 Jetson Transformers 与 Jetson TensorRT
Edge-LLM。服务器 GPU 的性能数值单独保存，不进入 Jetson 加速比。

Jetson Transformers FP16 如果无法加载、发生 OOM 或依赖不兼容，仍需生成失败记录；
该结果构成板端原生框架基线的一部分，不得省略后直接宣称 TensorRT 加速。

## 质量指标

- **JSON 有效率**：严格解析为 `ParkingAssessment` 的记录数 / 总记录数。
- **风险等级准确率**：预测 `risk_level` 与人工标注一致的记录数 / 总记录数；解析失败按错误计。
- **事件 micro precision/recall/F1**：在六类风险事件上累计 TP、FP、FN 后计算。
- **不安全建议率**：人工标注为 high risk，或包含行人/车辆接近行驶路径事件时，预测必须至少包含
  `yield` 或 `prepare_to_stop`；解析失败也计为不安全。
- **按事件错误**：每个事件分别统计 false positive 与 false negative。

当前安全策略是 `parking_risk_v1` 的保守规则。修改规则需要提升 schema/workload 版本，
不得用新规则重算并覆盖旧报告。

## 性能指标

- 只对成功记录聚合时延和资源值；失败继续保留在 `failure_summary` 和原始记录中。
- 每个实际采集到的阶段分别计算 p50、p90、p99。
- 首个成功执行的端到端时延记为 cold start。
- tokens/s 只在 runtime 同时报告 output tokens 与 decode latency 时计算。
- 峰值内存和峰值温度取最大值，平均功耗对已采集记录取算术平均。
- 未被 runtime 实际测量的指标保持 `null`，不使用估算值补齐。

## 失败类别

稳定类别包括输入错误、依赖不可用、JSON 解析失败、模型拒答、超时、内存不足、
不支持算子和其他 runtime 错误。每个 `InferenceRecord` 恰好包含 assessment 或
failure 之一，不会用默认业务输出掩盖失败。
