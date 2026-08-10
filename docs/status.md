# 当前实现状态

## 已实现并由无硬件测试覆盖

- `ParkingAssessment` 严格 JSON 解析、字段和枚举校验。
- `ParkingCaseCatalog` 的 manifest/annotation 一致性、来源组划分和图片引用校验。
- 冻结 workload 读取、字段约束和 SHA-256 身份。
- Transformers 与 TensorRT Edge-LLM HTTP 的 `RiskRuntime` Adapter seam。
- 成功、JSON 失败、超时、输入缺失等 `InferenceRecord` 事实记录。
- 质量指标、阶段时延分位数、资源汇总和 `StudyReport`。
- 从 `StudyReport` 与 `tegrastats` 原始日志派生后端完成率、端到端分位数和板端资源
  分位数的证据汇总入口。
- 单图分析、配置化研究和环境快照入口。
- LoRA 训练、合并、导出、engine 构建的可审计外部流程入口。

## 已形成的外部运行证据

- Jetson Orin Nano 的 L4T、CUDA、TensorRT、功耗模式、内存、swap 和磁盘已检查。
- `Qwen/Qwen3-VL-2B-Instruct` 已按不可变 commit
  `89644892e4d85e24eaac8bacfd4f463576704203` 缓存到 Jetson。
- 模型 snapshot 的文件数、总字节数、未完成文件和权重 SHA-256 已校验。
- 各实验与训练示例配置已统一固定到该模型 commit。
- x86 GPU 服务器已使用 TensorRT Edge-LLM `v0.9.1` 固定 commit 完成 LLM 与
  视觉编码器的 ONNX 导出；flow record 状态为 `succeeded`，全部声明输出存在。
- Jetson 已升级到 L4T R36.5.0 / JetPack 6.2.2，CUDA 12.6 与 TensorRT 10.3
  可用；ONNX 归档和内部 11 个文件均已重新校验。
- TensorRT Edge-LLM `v0.9.1` commit
  `7f061f21f0a581ba234a1e233c9315b89d8e47d6` 已在 Jetson 编译，插件、LLM/视觉
  builder、示例 runtime 与 Python binding 均存在，动态库依赖和 Python 导入通过。
- 视觉 FP16 engine 已在 Jetson 正式构建成功。`visual.engine` 为 786 MiB，SHA-256
  为 `3c6b4cce682e021b09c066d0e325335e31ef9edbf613c754be586035c26f5c2f`；flow
  record 状态为 `succeeded`，三个声明输出全部存在。
- 启用临时 8 GiB 磁盘 swap 后，LLM FP16 engine 已在 Jetson 构建成功。
  `llm.engine` 为 `3453798316` 字节，SHA-256 为
  `cbdf0300bf406dfbbcd06d47435c699c26403139d6bdd06b473ba00576583013`；flow
  record 状态为 `succeeded`，六个声明输出全部存在。
- 运行时补丁在创建 execution context 前调用 TensorRT
  `setWeightStreamingBudgetV2(0)`。LLM 与视觉 engine 已同时加载，HTTP `/health`
  返回 `healthy`，decode CUDA graph 捕获成功。
- 真实单图推理返回 HTTP 200 和 57 个 token，端到端 `38943.90 ms`；模型输出 JSON
  字段类型不符合严格 schema，因此如实记录为 `json_parse_error`。
- 冻结 `ps20_pilot_v1` 的 20 个样本均完成后端推理；严格 JSON 有效率为 `0/20`，
  失败汇总为 `json_parse_error: 20`。端到端 p50/p90/p99 分别为
  `41.76/50.61/55.42 s`，聚合端到端输出速率为 `1.48 token/s`。
- 874 条 `tegrastats` 记录显示 GPU 利用率均值 `98.79%`、RAM 峰值 `7414 MB`、
  swap 峰值 `2579 MB`、板端输入功耗均值 `10.11 W`、GPU 峰值温度 `62.5°C`。

上述状态证明固定版本的 `ONNX -> FP16 engine -> Edge-LLM HTTP -> ParkSight Adapter ->
InferenceRecord/StudyReport` 已在目标 Jetson 上真实执行。它不证明模型已经满足业务
质量要求：当前主要误差是 20/20 输出未遵守严格字段类型和领域事件枚举。
具体环境、命令和校验结果见 [`progress.md`](progress.md)。

## 尚未完成及证据边界

- Jetson Transformers FP16 在早期板端 smoke test 中 OOM，尚无同一 20 样本集上的
  成功报告，因此当前不能计算 Edge-LLM 相对 Transformers 的同机加速比。
- 服务器 Transformers 正确性参考尚未形成同一 `ps20_pilot_v1` 的完整 StudyReport。
- 当前数据只有 20 个冻结 `test` 样本，没有可用于训练的 `train`/`validation` 划分；
  LoRA 配置仍引用不存在的 `parking_risk_v1.jsonl`，训练命令仍是审核占位符。因此
  LoRA、合并模型及复测均明确记录为未执行。
- INT4 尚无经过审核的量化方法、校准集、构建配置或 engine；当前只有 FP16 的内存、
  时延和输出格式误差证据，不能声称 INT4 已完成。
- 临时 `/home/ubuntu/parksight-build.swap` 已启用但未写入 `fstab`；`Device or resource
  busy` 表示重复执行 `swapon`，不是启用失败。

完整命令、结果和原始证据索引见 [`execution-report.md`](execution-report.md)。
