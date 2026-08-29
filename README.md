# JetsonVLM

<p>
  <img src="https://img.shields.io/badge/Built%20with-Codex-412991" alt="Built with Codex">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Qwen3--VL-2B--Instruct-6E56CF" alt="Qwen3-VL 2B Instruct">
  <img src="https://img.shields.io/badge/TensorRT--Edge--LLM-v0.9.1-76B900" alt="TensorRT Edge-LLM v0.9.1">
  <img src="https://img.shields.io/badge/Transformers-server%205.9.0%20%7C%20Jetson%204.57.6-FFD21E" alt="Transformers versions">
  <img src="https://img.shields.io/badge/JetPack-6.2.2-76B900" alt="JetPack 6.2.2">
</p>

JetsonVLM 是一个面向低速泊车场景的视觉语言模型研究与部署项目，围绕
`Qwen/Qwen3-VL-2B-Instruct` 实现风险理解、领域 LoRA 适配、INT4 AWQ 量化、
TensorRT Edge-LLM 部署和可审计评测。

开发协作由 Codex 辅助完成代码、实验流程和文档整理；GPU 服务器与 Jetson
上的模型训练、量化、engine 构建和推理均保留独立命令、环境快照与运行证据。

## 关键版本

| 组件 | 版本或固定身份 | 用途 |
|---|---|---|
| Python | `>=3.10` | 项目运行环境 |
| Qwen3-VL | `Qwen/Qwen3-VL-2B-Instruct`，revision `89644892e4d85e24eaac8bacfd4f463576704203` | 基础视觉语言模型 |
| TensorRT Edge-LLM | `v0.9.1`，commit `7f061f21f0a581ba234a1e233c9315b89d8e47d6` | Jetson runtime、ONNX 导出和 engine 构建 |
| Transformers | 服务器 `5.9.0`；Jetson `4.57.6` | 训练、服务器质量参考和 Jetson FP16 基线 |
| PyTorch | 服务器 `2.8.0+cu128`；Jetson `2.9.1` | 模型训练与推理 |
| PEFT / Accelerate | `peft 0.18.0` / `accelerate 1.10.1` | LoRA 训练环境 |
| ModelOpt | `0.44.0` | INT4 AWQ 量化环境 |
| JetPack / CUDA / TensorRT | `6.2.2` / `12.6` / `10.3` | Jetson Orin Nano 板端环境 |

## 项目目标与输出

系统面向静态泊车场景图片，识别以下风险事件：

- `vru_near_maneuver_path`：行人等弱势交通参与者接近行驶路径
- `vehicle_near_maneuver_path`：车辆接近行驶路径
- `fixed_obstacle_near_path`：固定障碍物接近行驶路径
- `narrow_passage`：可用通行空间狭窄
- `visibility_occlusion`：可见性受到遮挡
- `parking_space_conflict`：车位或泊车空间存在冲突

输出结构由 `parking_risk_v1` 固定：

```json
{
  "schema_version": "parking_risk_v1",
  "risk_level": "medium",
  "events": ["narrow_passage"],
  "evidence": ["车辆两侧空间较窄，可用横向通行区域有限。"],
  "driver_advice": ["slow_down", "maintain_observation"]
}
```

其中：

- `risk_level` 是场景整体风险等级，只能是 `low`、`medium` 或 `high`。
- `events` 是从固定六类集合中选择的风险事件数组。
- `evidence` 是与图片可见线索对应的简短说明，不是思维链。
- `driver_advice` 是枚举式安全提示，不是车辆控制指令。

## 当前结论

部署链路已经完成从模型、ONNX、TensorRT engine、Edge-LLM HTTP 服务到
`InferenceRecord/StudyReport` 的闭环。当前主要问题是领域质量，而不是基础运行链路：

- Jetson Transformers FP16、Edge-LLM FP16、旧 LoRA FP16 和旧通用校准 INT4 均完成了冻结 20 样本评测。
- Edge-LLM Base FP16 的严格 JSON 有效率为 100%，风险等级准确率为 35%，事件 micro-F1 为 0.359。
- 旧 LoRA 在 Jetson 上的事件 micro-F1 为 0.389，但输出 token 数更少，不能将其较低端到端延迟直接解释为 runtime 加速。
- 旧通用校准 INT4 的端到端 p50 约 10.52 秒，但事件 micro-F1 退化为 0。
- 最新 16 条领域校准数据生成的 INT4 engine 运行完成 20/20，但严格 JSON 有效率只有 20%，主要失败模式是 Markdown `json` 代码围栏。
- 最新复核数据训练出的 LoRA adapter 在服务器冻结集上的风险准确率为 50%、事件 micro-F1 为 0.182；合并模型为 45% 和 0.100，尚未形成整体质量提升。
- 新增的 `parking_risk_v2_strict_json` 在保留 `parking_risk_v1` schema 的同时强化原始 JSON 边界；Jetson 领域 INT4 完整 20 样本 A/B 中，严格 JSON 有效率从 v1 的 20% 提升到 95%，但事件 micro-F1 仍为 0。

因此，当前优先级是扩大并人工终审领域标注、修正 LoRA 数据偏置、分析量化后的格式退化，再进行新模型的 Jetson 复测。

### 严格 JSON workload A/B

`parking_risk_v1` 是已有冻结 workload，`parking_risk_v2_strict_json` 是独立的提示词实验版本；两者的
`schema_version`、输入尺寸、生成参数、风险事件和驾驶建议枚举保持一致，因此可以把差异归因到提示词边界约束，
而不是 schema 或生成配置变化。v2 明确要求输出原始 JSON 对象，禁止 Markdown 代码围栏和前后说明；同时将重复的字段
说明压缩到 workload 渲染器追加的统一约束中，避免超过当前 `i768` engine 的最大输入长度。

`scripts/inspect_prompt_contract.py` 支持 `--max-input-tokens` 预算门禁。使用 Jetson 实际
Qwen3-VL processor 测得同一图片的 v1/v2 输入分别为 735/700 tokens，均满足 768 上限；
诊断报告中的 `prompt_tokens.count`、`prompt_tokens.budget` 和
`prompt_tokens.within_budget` 用于在启动 engine 前发现超长 workload。

在 Jetson 领域 INT4 engine `qwen3_vl_2b_int4_awq_ps16_v1_i768_k1024`、冻结的
`ps20_pilot_v1`、相同运行时参数下完成 20 样本 A/B；两次运行均为单次重复，v2 先运行、
v1 后运行：

| Workload | Workload identity | 后端完成 | 严格 JSON | 风险准确率 | 事件 micro-F1 | 端到端 p50 | 平均输出 tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| `parking_risk_v1` | `parking_risk_v1@sha256:8350ace4...a493` | 20/20 | 20% | 15% | 0 | 10.62 s | 80.8 |
| `parking_risk_v2_strict_json` | `parking_risk_v2_strict_json@sha256:c4695a1b...d1f4` | 20/20 | 95% | 35% | 0 | 7.43 s | 57.8 |

该 A/B 已覆盖完整冻结集，但仍是单次重复，不能单独作为稳定性能结论；v2 的格式有效率
明显提高，风险等级准确率也提高，但事件 micro-F1 没有改善，说明提示词修正尚未解决领域
识别质量。v1 报告见 `reports/jetson_edgellm_int4_awq_ps16_v1_ps20_pilot_i768_k1024.json`，
SHA-256 为 `4e7a3faa97ddcf27a68b20ef9df54b544419793ede06e0b77d83b9317b682bd6`；v2 报告见
`reports/jetson_edgellm_int4_awq_ps16_v1_ps20_pilot_strict_json_i768_k1024.json`，SHA-256 为
`9f75374756305d820baf8efd635a5ef709dc867a453e2632f426c78d897c1cc0`。两次服务日志分别为
`reports/jetson-int4-ps20-v1-20260830.log` 和 `reports/jetson-int4-ps20-strict-json-20260830.log`。
后续应在人工确认数据完成后，重新训练/量化并用同一完整 study 口径验收。

### 80 条 Codex 候选样本开发评测

为验证“先用 Codex 跑通识别与评测流程”的可行性，复用同一 Jetson INT4 engine 和
`parking_risk_v2_strict_json` workload，对 `ps80_development_v1` 的 64 条 train、16 条
validation 候选分片分别运行一次。参考标注来自 `ps80_reviewed_v1` 的 Codex 候选结果，
不属于人工终审金标，也不替代 `ps20_pilot_v1` 冻结测试集。

| 分片 | 后端完成 | 严格 JSON | 风险准确率 | 事件 micro-F1 | 不安全建议率 | 失败 |
|---|---:|---:|---:|---:|---:|---:|
| train（64） | 64/64 | 95.31% | 57.81% | 0 | 31.25% | `json_parse_error=3` |
| validation（16） | 16/16 | 100% | 62.50% | 0 | 18.75% | 无 |
| 合计（80） | 80/80 | 96.25% | 58.75% | 0 | 28.75% | `json_parse_error=3` |

这证明 80 条样本可以被当前识别流程完整处理并形成可审计记录，但当前模型几乎不输出
正确的风险事件，不能据此宣称领域识别能力达标。可复现实验配置为
`configs/studies/jetson_edgellm_int4_awq_ps16_ps80_codex_candidate_train_strict_json.json`
和对应的 validation 配置；原始报告保存在本地忽略目录 `reports/`，其 SHA-256 分别为
`21a3ef7dd7f0a4d76f0846a03448cca6c296b2458c0a1ec065b70a5958d10671` 和
`8edf7cd695cfffd919d521653bd7877421d470396705a12e089e76fb11ebb4dd`。

## 实验结果

### Jetson 同机运行时对比

以下结果均来自冻结的 `ps20_pilot_v1` 测试集，20 个样本、单次重复；性能只比较 Jetson 上的 runtime。服务器 GPU 的时延不用于推导 Jetson 加速比。

| Runtime / 模型 | 后端完成 | 严格 JSON | 风险准确率 | 事件 micro-F1 | 端到端 p50 | 聚合输出速率 |
|---|---:|---:|---:|---:|---:|---:|
| Jetson Transformers FP16 | 20/20 | 100% | 35% | 0.341 | 9.38 s | 未报告 |
| Edge-LLM Base FP16 | 20/20 | 100% | 35% | 0.359 | 50.75 s | 1.48 token/s |
| Edge-LLM 旧 LoRA FP16 | 20/20 | 100% | 35% | 0.389 | 30.60 s | 1.44 token/s |
| Edge-LLM 旧通用校准 INT4 | 20/20 | 100% | 35% | 0 | 10.52 s | 7.33 token/s |
| Edge-LLM 新领域校准 INT4 | 20/20 | 20% | 15% | 0 | 10.68 s | 7.32 token/s |

旧 LoRA 的端到端 p50 低于 Base FP16，但该轮输出 token 总数同时减少约 41.8%，聚合输出速率反而略低，因此该结果只能说明本轮完整输出更短，不能单独证明 LoRA runtime 更快。

### 服务器微调结果

服务器 Transformers 结果用于模型正确性、质量和误差分析，不与 Jetson 的端到端时延直接比较。

| 模型状态 | 严格 JSON | 风险准确率 | 事件 micro-F1 | 训练/推理说明 |
|---|---:|---:|---:|---|
| Base FP16/BF16 | 100% | 35% | 0.350 | 服务器质量参考 |
| 旧 `ps80` LoRA adapter | 100% | 35% | 0.389 | 1 epoch；旧弱监督数据 |
| 新 `ps64-reviewed` LoRA adapter | 100% | 50% | 0.182 | 3 epoch；non-low 样本过采样 2 倍 |
| 新 `ps64-reviewed` merged | 100% | 45% | 0.100 | 合并后结果与 adapter 不一致 |

新一轮 LoRA 使用 48 个唯一训练样本，过采样后为 63 条有效训练记录，训练 3 epoch、48 个 optimizer step，验证损失为 `0.7220`，峰值 CUDA 显存约 `5.273 GiB`，训练耗时约 `80.84 s`。

### INT4 量化结果

量化边界为 LLM backbone W4A16 AWQ、group size 128；visual、`lm_head` 和 KV cache 保持 FP16 或未量化。

| 量化版本 | 校准数据 | Engine | 严格 JSON | 事件 micro-F1 | 端到端 p50 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 旧通用文本校准 | 128 条新闻文本 | 约 1.36 GB | 100% | 0 | 10.52 s | 部署和格式通过，任务质量不通过 |
| 新领域校准 | 16 条泊车领域文本 | 约 1.36 GB | 20% | 0 | 10.68 s | 主要出现 Markdown JSON 围栏，质量不通过 |

相对 Edge-LLM Base FP16，旧 INT4 的平均端到端延迟缩短至约五分之一（约 5.02x 加速），engine 体积减少约 60.5%；但量化后的任务质量仍需单独验收，不能用速度或 engine 体积替代质量指标。

## 评测指标说明

### 严格 JSON 有效率

严格 JSON 有效率不是“模型返回了一段看起来像 JSON 的文本”的比例，而是：

```text
严格 JSON 有效率 = 能被解析并通过 ParkingAssessment 全部字段、枚举和数组校验的记录数 / 总记录数
```

以下情况都会失败：

- JSON 语法错误；
- 外层包含 Markdown 代码围栏或额外说明文字；
- 缺少字段或包含未知字段；
- `events`、`evidence`、`driver_advice` 类型错误；
- 风险等级、事件或驾驶建议不在固定枚举中。

“后端完成 20/20”只表示 runtime 得到了 20 个后端响应；“严格 JSON 20/20”才表示 20 个响应都能进入统一的领域对象。两者在报告中分开记录。

### 风险等级准确率

风险等级准确率比较整个样本的 `risk_level` 是否与人工标注一致：

```text
风险等级准确率 = risk_level 完全匹配的样本数 / 测试样本总数
```

它是样本级指标。例如，模型预测为 `medium`，但标注为 `high`，该样本即算错；即使模型正确识别了部分风险事件，也不会因此获得风险等级准确率分数。

### 事件 micro-F1

事件指标针对六类 `events` 做多标签集合比较。先在所有事件类别上累计 TP、FP、FN，再计算：

```text
micro-precision = TP / (TP + FP)
micro-recall    = TP / (TP + FN)
micro-F1        = 2 * precision * recall / (precision + recall)
```

因此，风险等级准确率和事件 micro-F1 衡量的是不同问题：前者判断场景整体等级，后者判断具体风险事件集合。模型可能风险等级判断正确，但漏掉 `visibility_occlusion`；也可能识别了某个事件，但整体风险等级判断错误。

项目同时记录按事件的 false positive 和 false negative，以及高风险或行人/车辆风险场景下的“不安全建议率”。解析失败按质量错误处理，不使用默认结果掩盖失败。

## 数据与实验身份

- 冻结测试集：`ps20_pilot_v1`，20 张图片。
- 开发数据：`ps80_development_v1`，80 个独立来源组。
- 复核数据拆分：48 条 LoRA train、16 条 validation、16 条独立 INT4 calibration。
- 数据拆分按 `source_group_id` 隔离，避免同一视频或连续采集序列跨 split 泄漏。
- 当前复核标注来源为 Codex 单轮视觉复核，不等同于人工双人金标。
- 基础模型 revision 固定为 `89644892e4d85e24eaac8bacfd4f463576704203`。
- TensorRT Edge-LLM 固定 commit 为 `7f061f21f0a581ba234a1e233c9315b89d8e47d6`。

一次可比较的 Study 同时固定 workload、manifest、annotation、split、backend revision、model revision、adapter revision、精度、功耗模式和重复次数。任一项变化都应生成新的 `study_id`，不覆盖旧报告。

## 项目结构

```text
src/parksight_vlm/
  assessment/       # ParkingAssessment 严格 JSON 解析和校验
  casebook/         # ParkingCase、manifest、annotation 和 split 校验
  inference/        # Transformers / TensorRT Edge-LLM Runtime Adapter
  studies/          # InferenceRecord、质量、性能和证据汇总
  app/              # 单图分析、批量 Study 和 runtime factory
configs/
  workloads/        # 冻结 prompt、生成参数、输入尺寸和 schema
  studies/          # 基线、LoRA、FP16、INT4 实验定义
  flows/            # 训练、合并、量化、导出和 engine 构建定义
data/
  manifests/        # 可提交的样本元数据
  annotations/      # 可提交的人工或复核标注
  raw/              # 本地图片输入，不进入 Git
artifacts/          # adapter、量化权重、ONNX 和 engine，本地生成
reports/            # StudyReport、flow record 和原始运行证据，本地生成
scripts/            # 训练、合并、量化、导出、构建和服务入口
tests/              # 无硬件单元测试
docs/               # 架构、数据、评测、部署和进展记录
```

主要 Module 接口：

- `ParkingAssessment.from_mapping(payload)`：解析并校验模型或标注 JSON。
- `ParkingCaseCatalog.load()` / `validate()`：加载样本并校验来源组、split 和图片引用。
- `RiskRuntime.analyze(case, workload)`：执行一次推理并保留成功或失败事实。
- `StudyRunner.run(casebook, runtime, study)`：聚合推理记录并生成 `StudyReport`。

## 80 样本候选标注复盘流程

项目支持先生成 Codex 候选标注，再由人工在同一份 package 上确认或修改。候选结果保持
`review_status=candidate`，不会自动进入最终金标状态：

```powershell
$env:PYTHONPATH = "src"
& ".\.venv\Scripts\python.exe" scripts\build_review_package.py `
  --manifest data\manifests\ps80_development_v1.jsonl `
  --candidate-annotations data\annotations\ps80_reviewed_v1.jsonl `
  --output reports\label-review-20260829\ps80_codex_review_package_v1.jsonl
```

联系表支持从嵌套的本地图片目录递归定位文件，并显示候选风险等级和事件。该命令需要
Pillow：

```powershell
& ".\.venv\Scripts\python.exe" scripts\build_label_review_sheets.py `
  --records data\manifests\ps80_development_v1.jsonl `
  --annotations data\annotations\ps80_reviewed_v1.jsonl `
  --image-root data\raw\ps2.0 `
  --output-directory reports\label-review-20260829
```

复盘过程中可随时检查完成度；该命令只读 package，并用 `--fail-on-incomplete` 在仍有
待处理或非法定稿记录时返回退出码 `2`：

```powershell
& ".\.venv\Scripts\python.exe" scripts\inspect_review_package.py `
  --package reports\label-review-20260829\ps80_codex_review_package_v1.jsonl `
  --fail-on-incomplete
```

复盘完成后，再使用 `scripts/prepare_reviewed_lora_dataset.py` 校验 schema、来源组隔离、
LoRA train/validation 与 INT4 calibration 拆分。当前 package 仍属于候选标注，人工确认
结果应写入 `human_assessment`，并将 `review_status` 改为 `confirmed` 或 `corrected`；说明
写入 `review_note`。确认后先运行：

```powershell
& ".\.venv\Scripts\python.exe" scripts\finalize_review_package.py `
  --package reports\label-review-20260829\ps80_codex_review_package_v1.jsonl `
  --annotations-output data\annotations\ps80_human_confirmed_v1.jsonl `
  --summary-output reports\label-review-20260829\ps80_human_confirmed_v1_summary.json
```

生成的标准 annotation JSONL 再作为 `prepare_reviewed_lora_dataset.py` 的
`--annotations` 输入。

进入训练或校准数据生成时必须显式声明标注来源，例如人工终审后的数据使用
`--label-source human_confirmed_v1`；候选数据则保留
`--label-source codex_visual_review_v1_single_pass`。生成的 LoRA/校准记录和 summary
会沿用该来源标识。

正式 LoRA 训练入口还会再次核对配置与数据集中的 `label_source`，并默认拒绝
`codex_visual_review_v1_single_pass`。因此，当前候选数据即使已经完成 JSON schema 和
split 校验，也不会被误用于正式训练；只有将 package 定稿为 `confirmed`/`corrected`、
生成 `human_confirmed_v1` 数据并更新训练配置后，训练流程才会继续。

如果需要在人工终审前验证训练链路，可显式使用
`configs/flows/train_qwen3_vl_2b_lora_ps64_codex_candidate_v1.json`。该配置只允许候选
来源进入显式命名的 `data/processed/lora/ps64_codex_candidate_v1.jsonl` 数据和
`codex_candidate` 实验产物，训练摘要和后续报告不得作为人工金标或最终
质量结论；训练完成后可使用
`configs/studies/server_transformers_lora_ps64_codex_candidate_v1_ps20_pilot.json` 做服务器
冻结集验证。正式复现仍使用 `train_qwen3_vl_2b_lora_ps64_reviewed_v1.json`。

定稿脚本会同时输出 `candidate_human_comparison`，包括候选风险等级准确率、事件
micro-precision、micro-recall、micro-F1，以及整体标注、风险等级、事件集合、证据和
驾驶建议的修正数量。`confirmed` 要求人工结果与候选结果完全一致；`corrected` 要求
结果确实发生变化，并必须填写 `review_note`。因此，80 条样本的识别效果应以人工确认
后的 comparison 指标为准，而不是以候选 package 的数量或 JSON schema 通过率代替。

## 开始使用

无硬件测试：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

外部模型、GPU、Jetson 和 engine 操作均通过显式配置和命令进入，不在导入项目或运行无硬件测试时自动下载模型、启动服务或构建 engine。

相关文档：

- [当前实现状态](docs/status.md)
- [项目进展与环境记录](docs/progress.md)
- [完整执行记录](docs/record.md)
- [系统架构](docs/architecture.md)
- [评测口径](docs/evaluation.md)
- [操作入口](docs/operations.md)
- [领域术语](CONTEXT.md)
- [项目学习记录](docs/personal_record.md)
