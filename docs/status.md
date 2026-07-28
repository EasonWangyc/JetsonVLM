# 当前实现状态

## 已实现并由无硬件测试覆盖

- `ParkingAssessment` 严格 JSON 解析、字段和枚举校验。
- `ParkingCaseCatalog` 的 manifest/annotation 一致性、来源组划分和图片引用校验。
- 冻结 workload 读取、字段约束和 SHA-256 身份。
- Transformers 与 TensorRT Edge-LLM HTTP 的 `RiskRuntime` Adapter seam。
- 成功、JSON 失败、超时、输入缺失等 `InferenceRecord` 事实记录。
- 质量指标、阶段时延分位数、资源汇总和 `StudyReport`。
- 单图分析、配置化研究和环境快照入口。
- LoRA 训练、合并、导出、engine 构建的可审计外部流程入口。

## 需要外部数据或硬件才能形成证据

- 冻结数据集、人工标注与来源组审计。
- 服务器 Transformers 正确性参考与完整质量研究。
- Jetson 板端 Transformers FP16 实际运行与报告。
- Jetson 上 TensorRT Edge-LLM FP16 engine 构建与推理。
- 领域 LoRA 训练、合并模型复测与误差分析。
- LLM backbone INT4 的校准、构建和同口径研究。
- 板端冷启动、分阶段时延、吞吐、峰值内存、功耗、温度和失败样例。

这些项目不是代码完成度声明。对应报告只有在实际模型、数据和目标设备上执行后才会
从“待验证”转为“有实测证据”。

## 已完成的外部准备

- Jetson Orin Nano 环境、CUDA 可用性、功耗模式、内存和磁盘已检查。
- `Qwen/Qwen3-VL-2B-Instruct` 已按不可变 commit
  `89644892e4d85e24eaac8bacfd4f463576704203` 缓存到 Jetson。
- 模型 snapshot 的文件数、总字节数、未完成文件和权重 SHA-256 已校验。
- 各实验与训练示例配置已统一固定到该模型 commit。

上述状态说明环境和模型文件已经就绪，不代表真实单图推理或板端基线已经完成。
具体环境、命令和校验结果见 [`progress.md`](progress.md)。

## 当前板端阻塞

2026-07-28 已建立隔离的 Python 3.10 Jetson runtime。PyTorch 2.9.1 的
`torch.cuda.get_arch_list()` 返回 `['sm_87']`，小 tensor CUDA kernel 和 29 个
无硬件测试均通过，原通用 aarch64 wheel 的架构不兼容问题已经解决。

真实 FP16 模型加载仍出现 `NvMapMemAllocInternalTagged ... error 12`。关闭 PyTorch
CUDA caching allocator 后仍然复现；失败时系统总可用内存尚未耗尽，但 lfb 只剩
2×4 MiB，GPU 计算尚未开始。该结果与 L4T R36.4.7 的已知 NvMap 问题一致，当前不能
据此断言 2B FP16 模型本身无法装入 8GB。

下一步是评估升级到包含 r36.5 的 JetPack 6.2.2，升级后按相同 workload 重跑。
完整命令、结果和原始证据索引见 [`execution-report.md`](execution-report.md)。
