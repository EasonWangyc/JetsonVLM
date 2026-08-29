# 当前实现状态

## 2026-08-30 严格 JSON workload A/B checkpoint

- 新增 `configs/workloads/parking_risk_v2_strict_json.json` 和对应的
  `jetson_edgellm_int4_awq_ps16_v1_ps20_pilot_strict_json` study。v2 保留
  `parking_risk_v1` schema、输入尺寸、生成参数和枚举集合，仅强化原始 JSON 输出边界，
  identity 为 `parking_risk_v2_strict_json@sha256:c4695a1bfa4d547f5ad90ec7697b82419dad12995829776e96c850707e15d1f4`。
- 在 Jetson 同一领域 INT4 engine、同一 `ps20_pilot_v1` 和相同运行时参数下完成 v1/v2
  完整 A/B。v1 为 `20/20` 后端完成、严格 JSON `20%`、风险准确率 `15%`、事件
  micro-F1 `0`、端到端 p50 `10618 ms`；v2 为 `20/20`、`95%`、`35%`、`0`、
  p50 `7428 ms`。v1 的 16 条失败均为 `json_parse_error`，v2 仅 1 条失败。
- v2 之前的单图 smoke 也返回 HTTP 200、严格 JSON 解析成功；完整 study 进一步证明格式
  修正不是单样本偶然现象，但风险事件识别质量仍未改善。
- 第一次 v2 请求曾因提示词达到 823 token、超过 `i768` engine 的 768 token 输入上限而
  返回 HTTP 500；压缩重复约束后重新验证通过。该事实纳入 workload 设计约束，后续完整
  study 需要同时报告输入 token 预算和格式有效率。
- prompt-contract 诊断已增加 token 计数和 `--max-input-tokens` 门禁；Jetson 实际 Qwen3-VL
  processor 测得固定 `001.jpg` 的 v1/v2 输入为 `735/700` tokens，均返回
  `within_budget=true`，并确认 `messages_equal=true`。
- 本次服务已停止；日志归档于
  `reports/jetson-int4-strict-json-smoke-20260830.log`，SHA-256 为
  `6850bee6589a026cd4e2a24cb6d7e8e5e090341bcf0ee94ac4619cfd577af5a1`。
- 完整 v1 StudyReport 归档于
  `reports/jetson_edgellm_int4_awq_ps16_v1_ps20_pilot_i768_k1024.json`，SHA-256 为
  `4e7a3faa97ddcf27a68b20ef9df54b544419793ede06e0b77d83b9317b682bd6`；完整 v2 StudyReport
  归档于 `reports/jetson_edgellm_int4_awq_ps16_v1_ps20_pilot_strict_json_i768_k1024.json`，
  SHA-256 为 `9f75374756305d820baf8efd635a5ef709dc867a453e2632f426c78d897c1cc0`。两次服务
  均已停止。
- 复用同一 engine 对 80 条 Codex 候选开发样本完成一次 train/validation 分片评测：
  `64+16` 条后端全部完成，严格 JSON `77/80=96.25%`，候选标签上的风险准确率
  `47/80=58.75%`，事件 micro-F1 `0`，不安全建议率 `23/80=28.75%`，共有 3 条
  `json_parse_error`。该结果只用于开发流程和错误分析，不是人工确认后的质量结论。
- 对应配置为
  `jetson_edgellm_int4_awq_ps16_ps80_codex_candidate_train_strict_json` 与 validation
  版本；报告 SHA-256 分别为
  `21a3ef7dd7f0a4d76f0846a03448cca6c296b2458c0a1ec065b70a5958d10671` 和
  `8edf7cd695cfffd919d521653bd7877421d470396705a12e089e76fb11ebb4dd`。评测服务已停止。
- 已在本地 RTX 4060 隔离训练环境完成候选 LoRA：3 epochs、48 个唯一训练样本、63 条
  过采样记录、48 个 optimizer steps，validation loss `0.72318`，峰值显存 `5.272 GiB`，
  训练耗时约 `146 s`。候选 adapter 在冻结 `ps20_pilot_v1` 上为风险准确率 `50%`、
  事件 micro-F1 `0.1818`；同环境 base 为 `35%`、`0.350`，事件识别反而下降。
- 候选 adapter 已成功 merge，merged checkpoint 与 adapter 的 20 条输出逐样本一致，
  对应配置为 `merge_qwen3_vl_2b_lora_ps64_codex_candidate_v1` 和
  `server_transformers_merged_lora_ps64_codex_candidate_v1_ps20_pilot`。该实验仍使用
  Codex 候选标签，不构成正式质量结论。

## 2026-08-29 现场复核 checkpoint

- 本地仓库已提交 Codex-assisted review workflow、训练 provenance 防误用和部署预检改动；
  当前 `main` 比 `origin/main` 超前 13 个本地提交，未执行 push。
- 无硬件测试为 `61/61` 通过；80 条 Codex 候选 package 可由
  `build_review_package.py` 重新生成，当前候选分布为 `low=47`、`medium=32`、
  `high=1`。
- 已使用真实候选 annotation 完成一次数据生成 dry-run/校验：LoRA 共 64 条（48 train、
  16 validation），INT4 calibration 共 16 条；LoRA 与 calibration 来源组交集为 0，
  开发数据与冻结测试集来源组交集为 0。输出 summary 同时固定了 workload identity、
  manifest、teacher annotation、candidate annotation 和 calibration config 的 SHA-256。
- 定稿脚本现在会校验候选和人工两份严格 `ParkingAssessment`，强制
  `confirmed`/`corrected` 状态与内容一致，并输出候选相对人工结果的风险准确率、事件
  micro-F1 和字段修正统计。当前没有人工确认 package，因此不能把候选结果写成最终
  质量结论。
- 正式 LoRA 训练入口现在要求训练配置声明 `label_source`，并核对所有数据记录的来源
  一致性；候选 Codex 来源默认被拒绝。当前 `ps64_reviewed_v1` 配置因此会在启动早期
  停止，等待人工终审后的 `human_confirmed_v1` 数据。
- 为支持“先跑通 Codex 开发流程、后做人工复盘”，新增独立的
  `train_qwen3_vl_2b_lora_ps64_codex_candidate_v1.json`；它显式允许候选标签，但输出
  数据 `data/processed/lora/ps64_codex_candidate_v1.jsonl`、输出目录和 flow_id 均带
  `codex_candidate`，summary 的 `dataset_id` 为 `ps80_codex_candidate_v1`，与正式训练
  配置隔离。
- 新增 `inspect_review_package.py` 只读检查入口，可输出复核状态计数、待处理 case_id、
  候选风险/事件分布和 `ready_for_finalize`；使用 `--fail-on-incomplete` 可将未完成复核
  作为流程失败处理。
- Edge-LLM 服务入口新增 `--check-only` 静态预检；它已在 Jetson 现有 FP16、LoRA FP16、
  普通 INT4、领域 INT4 四种 engine 组合上完成检查，可区分文件/路径就绪与实际 GPU
  加载成功。
- 服务入口同时支持 `--llm-engine-root` 与 `--visual-engine-root` 分目录模式，覆盖
  LoRA/INT4 LLM engine 复用 FP16 visual engine 的实际部署结构。
- 已通过 SSH 只读连接 Jetson。板端工作树为旧提交 `f362a43` 且存在 33 项未提交/未跟踪
  改动，本轮未覆盖或清理。临时补充 venv 内 CUDA 库路径后，PyTorch `2.9.1`、CUDA
  `12.6` 和 Transformers `4.57.6` 可导入。
- 板端 HTTP 服务当前未运行。使用明确的 Edge-LLM 源码、pybind 和插件路径后，FP16 LLM
  engine、tokenizer 和 base context 可以加载，但图形桌面状态下 visual engine 申请约
  `811 MiB` 连续内存失败。该结果属于资源条件失败，不改变既有 headless 条件下的
  `20/20` FP16 成功证据。
- 2026-08-30 对领域 INT4 做了新的真实 smoke：去掉未启用的 weight-streaming 参数，
  显式加入 Jetson venv `site-packages` 后，LLM/visual engine、tokenizer、CUDA graph 和
  Uvicorn 均成功启动，`/health` 返回 HTTP 200；1 条单图请求也返回 HTTP 200 和 91 tokens，
  但严格 JSON 因 ```json 代码围栏解析失败。临时服务已停止，原始日志归档于
  `reports/jetson-int4-smoke-20260830.log`，SHA-256 为
  `9ee94b7b331368c7e7204288938eadd1bdd3f81e05e0c97209210bd7d77534b5`。

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

### 候选标注复核工具状态

80 条开发样本已经可以通过 `build_review_package.py` 固化为 Codex 候选 package，并在
人工复盘后通过 `finalize_review_package.py` 转换为标准 annotation JSONL。定稿过程现在
会校验候选和人工两份 `ParkingAssessment`、强制 `confirmed`/`corrected` 状态与内容一致，
并输出候选相对人工结果的风险准确率、事件 micro-F1 和字段修正统计。当前实际 package
仍是 `review_status=candidate`，因此尚未产生人工金标，也不能把候选结果写成最终质量结论。

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
