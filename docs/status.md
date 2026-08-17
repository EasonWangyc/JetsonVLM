# 当前实现状态

## 已实现并由无硬件测试覆盖

- `ParkingAssessment` 严格 JSON 解析、字段和枚举校验。
- `ParkingCaseCatalog` 的 manifest/annotation 一致性、来源组划分和图片引用校验。
- 冻结 workload 读取、字段约束和 SHA-256 身份。
- Transformers 与 TensorRT Edge-LLM HTTP 的 `RiskRuntime` Adapter seam。
- Transformers 可选 forward-hook profiling，可独立记录视觉编码、prefill 和 decode；
  插桩 study 与未插桩性能基线使用不同 `study_id`。
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
- Jetson Transformers FP16 已在冻结 `ps20_pilot_v1` 上完成 20/20 推理，严格 JSON
  有效率为 100%，失败汇总为空；未插桩端到端 p50/p90/p99 为
  `9.38/14.12/27.94 s`，进程 CUDA 峰值为 `4176.72 MB`。
- 独立插桩 study 同样 20/20 成功；阶段 p50 为预处理 `27.60 ms`、视觉编码
  `225.11 ms`、prefill `626.51 ms`、decode `17.04 s`。插桩强制 CUDA 同步，绝对
  时延不与未插桩基线混用；decode 占插桩生成 p50 的约 `88.5%`。
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
- 多模态 system content 已统一为 Qwen3-VL chat template 接受的 content 数组，并由
  固定 fixture 和 40 个无硬件测试覆盖。
- Edge-LLM C++ runtime 已应用“先分配 base/decoder context，再加载视觉 runner”的
  补丁；最终 FP16 LLM engine profile 为 input 768、KV 1024，单图连续 3/3 严格 JSON
  验收通过。
- 冻结 `ps20_pilot_v1` 的 20 个样本均完成后端推理；严格 JSON 有效率为 `20/20`，
  失败汇总为空。端到端 p50/p90/p99 分别为 `50.75/69.08/75.08 s`，聚合端到端
  输出速率为 `1.48 token/s`。
- 质量结果为风险等级准确率 `35%`、事件 micro-F1 `0.359`、不安全建议率 `0%`。
- 542 条 `tegrastats` 记录显示 GPU 利用率均值 `97.39%`、RAM 峰值 `7418 MB`、
  swap 峰值 `1904 MB`、板端输入功耗均值 `10.05 W`、GPU 峰值温度 `65.03°C`。
- 从 PS2.0 `training` 选取的 80 个独立来源组已生成联系表并完成 Codex 单轮视觉复核
  候选标注。相对 teacher 标签，33 条风险等级和 77 条事件集合被修正；当前拆分为
  48 条 LoRA train、16 条 validation 和 16 条独立 INT4 calibration，三者与 20 张
  pilot 冻结测试集均无来源组交集。该版本不是人工双人金标，仍需人工终审。
- RTX 4090 D 已完成 Qwen3-VL-2B LoRA：语言 attention 的 `q/k/v/o_proj` 共
  `6422528` 个可训练参数，占总参数 `0.301%`；1 epoch、16 个 optimizer step，
  验证损失 `0.0845`，峰值 CUDA 显存 `5.25 GiB`，训练用时 `21.54 s`。
- 同一服务器冻结测试集上，Base/LoRA 严格 JSON 均为 20/20，事件 micro-F1 从
  `0.3500` 提升至 `0.3889`，`vehicle_near_maneuver_path` 假阳性由 6 降为 0；风险
  准确率仍为 35%，`visibility_occlusion` 仍全部漏检。
- LoRA adapter 已合并为独立 checkpoint，合并模型复测质量指标与 adapter 一致；
  TensorRT Edge-LLM ONNX 导出 flow 状态为 `succeeded`。
- LoRA 合并模型的 LLM ONNX 归档与内部 7 个文件已在 Jetson 复算 SHA-256；新的
  FP16 weight-streaming LLM engine 构建 flow 状态为 `succeeded`，engine 为
  `3453786212` 字节，SHA-256 为
  `d38adc5d532615d7183a6b4aa8413020bd76a5991e49ecd51ca84d0442334224`。
- Jetson LoRA runtime 在 headless 模式和内存 compaction 后完成双 engine 加载、HTTP
  健康检查和同图连续 3/3 严格 JSON 验收；冻结 20 样本 20/20 完成，严格 JSON
  有效率为 100%，风险准确率为 35%，事件 micro-F1 从板端 Base FP16 的 `0.3590`
  提升至 `0.3889`。
- Jetson LoRA 端到端 p50/p90/p99 为 `30.60/33.08/40.90 s`，聚合输出速率为
  `1.44 token/s`；319 条遥测记录显示 RAM 峰值 `7351 MB`、swap 峰值 `580 MB`、
  GPU 利用率均值 `98.20%`、输入功耗均值 `10.10 W`、GPU 峰温 `61.94°C`。本轮输出
  token 总数比 Base 少 41.8%，因此延迟下降不能解释为 LoRA runtime 加速。
- Jetson INT4 AWQ 20 样本 20/20 完成且 JSON 有效率为 100%；相对 Edge-LLM FP16，
  平均延迟获得 `5.02x` 加速、输出速率获得 `4.95x` 提升、engine 减少 `60.5%`、
  RAM 峰值降低 `31.6%`。但事件 micro-F1 从 `0.359` 降为 0，明确记录为校准质量退化。
- 复核候选数据已完成三轮 LoRA 对照。最终 non-low x2、3 epoch adapter 在服务器冻结
  20 样本上严格 JSON 为 100%、风险准确率为 `50%`、事件 micro-F1 为 `0.182`；
  相比未平衡训练避免了全 low 塌缩，但事件 F1 低于旧弱监督 LoRA 的 `0.389`。
- 独立 ps16 领域文本已完成 LLM backbone W4A16 AWQ、ONNX 导出和 Jetson engine
  构建。新 engine SHA-256 为
  `589d8ba247a93cdf794c86697bb5a5d5fe3387fee812744c51d09806912b3026`，构建 flow
  状态为 `succeeded`。
- 新领域 INT4 在 Jetson 上 20/20 后端完成，但严格 JSON 仅 `4/20`，16 条
  `json_parse_error` 主要来自 Markdown JSON 围栏；风险准确率为 `15%`、事件
  micro-F1 为 0。端到端 p50 为 `10.68 s`、聚合输出速率为 `7.32 token/s`，性能与
  旧通用校准 INT4 相近，但质量更差，明确记录为负向实验。

上述状态证明固定版本的 Transformers FP16、Edge-LLM Base FP16、LoRA 合并模型和
INT4 均已完成相应的
`ONNX -> engine -> Edge-LLM HTTP -> ParkSight Adapter -> InferenceRecord/StudyReport`
板端实测。它不证明模型已经满足业务质量要求：当前主要误差已经从输出格式转为风险
等级和领域事件判断，仍需扩大独立人工标注数据并修正 LoRA 与量化校准偏置。
具体环境、命令和校验结果见 [`progress.md`](progress.md)。

## 尚未完成及证据边界

- Transformers FP16 早期在旧环境/低连续内存状态下有明确 OOM 记录；当前
  R36.5.0、图形桌面和启用 swap 的环境中，固定模型完整映射到 `cuda:0` 并完成
  20 样本基线与 20 样本插桩 study。插桩期间系统 RAM 峰值 `7302/7619 MB`、swap
  峰值 `1174 MB`、最小 lfb 为 `1x2 MB`，说明当前配置可运行但内存余量很小。
- Base FP16/LoRA LLM engine 在图形桌面状态下均可能因 NvMap 无法分配约 811 MB
  视觉 engine 内存而 OOM；本轮 LoRA 实测通过临时切换 `multi-user.target`、释放显示栈
  NvMap 客户端并执行内存 compaction 完成。评测结束后已恢复图形桌面与系统服务。
- 已建立修正弱监督偏置的 80 条视觉复核候选标注并完成 LoRA 重训，但仍是 Codex 单轮
  复核，不是人工双人金标；新 LoRA 的事件 F1 低于旧实验，不能描述为整体质量提升。
- 已使用 16 条无泄漏泊车领域文本完成 INT4 重新量化和板端复测；格式有效率退化到
  20%，说明小规模领域校准没有通过质量验收。后续应先扩充并人工终审校准数据，而不是
  用宽松解析器掩盖 Markdown 围栏问题。
- 临时 `/home/ubuntu/parksight-build.swap` 已启用但未写入 `fstab`；`Device or resource
  busy` 表示重复执行 `swapon`，不是启用失败。

完整命令、结果和原始证据索引见 [`execution-report.md`](execution-report.md)。
