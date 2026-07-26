# 操作入口

## 1. 无硬件验证

仓库的单元测试不加载模型权重，也不要求 CUDA、摄像头或 Jetson：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall -q src tests scripts
git diff --check
```

安装为开发包后，也可以直接使用 `parksight-analyze` 与 `parksight-study` 两个命令。

## 2. 单图分析

当前模型配置固定到 Hugging Face 不可变 commit
`89644892e4d85e24eaac8bacfd4f463576704203`。单图执行示例：

```bash
PYTHONPATH=src python3 -m parksight_vlm.app.analyze_image \
  --image data/raw/example.jpg \
  --runtime transformers \
  --backend-revision REPLACE_WITH_TRANSFORMERS_VERSION \
  --model-revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --precision bf16
```

命令输出完整 `InferenceRecord`。成功时包含严格校验后的 `assessment`，失败时包含
明确的失败类别和错误事实。单图分析不要求参考标注。

TensorRT Edge-LLM HTTP Adapter 使用实验性 OpenAI-compatible server：

```bash
PYTHONPATH=src python3 -m parksight_vlm.app.analyze_image \
  --image data/raw/example.jpg \
  --runtime tensorrt_edge_llm_http \
  --backend-revision REPLACE_WITH_EDGE_LLM_COMMIT \
  --model-revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --precision fp16 \
  --edge-url http://127.0.0.1:8000
```

HTTP server 的启动方式、请求兼容性和目标模型支持情况必须以 Jetson 上实际安装的
TensorRT Edge-LLM revision 为准。

## 3. 冻结研究

1. 按 [`data.md`](data.md) 准备 manifest、人工标注和图片。
2. 确认 `configs/studies/*.json` 中的 `model_revision` 与实际缓存和待构建 engine
   使用同一个不可变 commit；升级模型时必须显式修改并重新验证。
3. 检查 backend、精度、power mode 和输出路径。
4. 为三类实验使用独立配置和报告：

| 配置 | 运行位置 | 状态 |
| --- | --- | --- |
| `configs/studies/transformers_base.json` | GPU 服务器 | 已提供，作为正确性参考 |
| `configs/studies/jetson_transformers_fp16.json` | Jetson | 已提供，作为板端 Transformers FP16 基线 |
| `configs/studies/edgellm_fp16.json` | Jetson | 已提供，作为 TensorRT Edge-LLM FP16 基线 |

服务器正确性参考：

```bash
PYTHONPATH=src python3 -m parksight_vlm.app.run_study \
  --config configs/studies/transformers_base.json
```

Jetson Transformers FP16：

```bash
PYTHONPATH=src python3 -m parksight_vlm.app.run_study \
  --config configs/studies/jetson_transformers_fp16.json
```

Jetson TensorRT Edge-LLM FP16：

```bash
PYTHONPATH=src python3 -m parksight_vlm.app.run_study \
  --config configs/studies/edgellm_fp16.json
```

研究入口在运行前校验来源组划分和标注完整性。报告保存配置身份、环境快照、每条
`InferenceRecord`、质量指标、性能分位数和失败汇总。

三类配置必须使用相同的 workload、模型 revision 和冻结测试集。服务器报告用于正确性
参考；Jetson Transformers 与 Jetson Edge-LLM 报告用于同机性能比较。

## 4. LoRA、合并、导出与 engine 构建

`configs/flows/*.example.json` 是待审核模板。先复制为不带 `.example` 的实际配置，
填入当前环境中已确认的命令、输入和输出，再做 dry-run：

```bash
PYTHONPATH=src python3 scripts/train_lora.py \
  --config configs/flows/train_lora.json
PYTHONPATH=src python3 scripts/merge_lora.py \
  --config configs/flows/merge_lora.json
PYTHONPATH=src python3 scripts/export_model.py \
  --config configs/flows/export_model.json
PYTHONPATH=src python3 scripts/build_engine.py \
  --config configs/flows/build_engine.json
```

默认行为只输出 readiness，不执行外部命令。确认以下条件后，才增加 `--execute`：

- 命令和版本已经人工复核；
- required inputs 全部存在；
- expected outputs 尚不存在；
- record 与 log 路径尚未被占用。

执行时命令参数直接传给子进程，不经过 shell。结果只有在进程返回 0 且全部预期输出
存在时才记为 `succeeded`；日志和结构化结果均进入 `reports/flows/`。

## 5. 证据边界

代码可用、命令可启动和模型完成部署是三种不同状态。报告中只有实际执行产生的记录
可以作为质量、时延、内存、功耗或温度证据。dry-run、engine 文件存在、README
描述以及服务器可连接均不能替代板端推理验证。
