# TensorRT Edge-LLM FP16 部署流程

本文记录 `Qwen/Qwen3-VL-2B-Instruct` 的首个 TensorRT Edge-LLM
板端实验流程。命令尚未执行，只有生成的 flow record、engine、单图推理记录和
`StudyReport` 才属于部署证据。

## 1. 固定实验身份

| 项目 | 固定值 |
| --- | --- |
| 模型 | `Qwen/Qwen3-VL-2B-Instruct` |
| 模型 revision | `89644892e4d85e24eaac8bacfd4f463576704203` |
| TensorRT Edge-LLM revision | `7f061f21f0a581ba234a1e233c9315b89d8e47d6` (`v0.9.1`) |
| 精度 | LLM FP16、视觉编码器 FP16 |
| Jetson 平台 | Orin Nano、JetPack 6.2.2、CUDA 12.6 |
| 工作负载 | `configs/workloads/parking_risk_v1.json` |
| 板端对照配置 | `configs/studies/jetson_transformers_fp16_ps20_pilot.json` |
| Edge-LLM 配置 | `configs/studies/jetson_edgellm_fp16_ps20_pilot.json` |

两个板端 Study 必须使用相同的模型 revision、工作负载、PS2.0 样本、功耗模式和
生成上限。Edge-LLM 的 engine 只服务于上述 Edge-LLM revision、TensorRT 版本和
构建参数；任一项变化都重新构建，不复用旧 engine。

## 2. 产物边界

```text
GPU 服务器
models/Qwen3-VL-2B-Instruct-89644892/
  -> artifacts/onnx/qwen3_vl_2b_fp16/{llm,visual}/

Jetson
artifacts/onnx/qwen3_vl_2b_fp16/{llm,visual}/
  -> artifacts/engines/qwen3_vl_2b_fp16/{llm,visual}/
  -> Edge-LLM HTTP server
  -> InferenceRecord / StudyReport
```

模型下载与 ONNX 导出在 x86 Linux NVIDIA GPU 服务器执行；C++ runtime 编译、
TensorRT engine 构建和推理在 Jetson 执行。不要在 Jetson 上使用 server 的
`--model` 参数，因为这会把模型解析和 ONNX 导出重新带回板端。

## 3. GPU 服务器：固定源码与导出环境

以下命令在 GPU 服务器 Bash 中逐条执行。先确认服务器 CUDA、GPU 和磁盘：

```bash
nvidia-smi
nvcc --version
python3 --version
df -h
```

成功标准：x86-64 Linux、NVIDIA GPU、CUDA 12.x 或 13.x、Python 3.10+，并为模型、
ONNX 和临时文件保留足够空间。

固定 TensorRT Edge-LLM 源码：

```bash
cd /path/to/workspace
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git checkout --detach 7f061f21f0a581ba234a1e233c9315b89d8e47d6
git submodule update --init --recursive
git rev-parse HEAD
```

最后一条命令必须输出
`7f061f21f0a581ba234a1e233c9315b89d8e47d6`。

在独立导出环境中安装该 checkout 的依赖：

```bash
python3 -m venv .venv-export
source .venv-export/bin/activate
python -m pip install --upgrade pip
python -m pip install .
tensorrt-edgellm-export --help
```

在服务器上的 ParkSight-VLM 仓库下载固定模型 revision。若已经存在模型目录，先用
文件清单和 Hugging Face 缓存信息核对 revision，不覆盖未知来源目录。

```bash
cd /path/to/JetsonVLM
hf download Qwen/Qwen3-VL-2B-Instruct \
  --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --local-dir models/Qwen3-VL-2B-Instruct-89644892
```

先 dry-run，再显式导出：

```bash
PYTHONPATH=src python3 scripts/export_model.py \
  --config configs/flows/export_qwen3_vl_2b_fp16.json

PYTHONPATH=src python3 scripts/export_model.py \
  --config configs/flows/export_qwen3_vl_2b_fp16.json \
  --execute
```

成功标准：命令返回 0，并同时生成：

- `artifacts/onnx/qwen3_vl_2b_fp16/llm/model.onnx`
- `artifacts/onnx/qwen3_vl_2b_fp16/visual/model.onnx`
- 两个目录对应的 `config.json`，以及 LLM tokenizer/chat template、视觉预处理配置
- `reports/flows/export_qwen3_vl_2b_fp16.json`
- `reports/flows/export_qwen3_vl_2b_fp16.log`

## 4. 将 ONNX 传到 Jetson

将完整的 `artifacts/onnx/qwen3_vl_2b_fp16` 目录复制到 Jetson 的
`/home/ubuntu/JetsonVLM/artifacts/onnx/`。如果租赁服务器不能直接访问
`192.168.137.187`，使用 PC 做两段式中转，不改变目录结构。

传输前后分别执行：

```bash
find artifacts/onnx/qwen3_vl_2b_fp16 -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > qwen3_vl_2b_fp16.sha256

sha256sum -c qwen3_vl_2b_fp16.sha256
```

成功标准：Jetson 上全部文件校验为 `OK`。

## 5. Jetson：编译固定版本的 C++ runtime

以下命令在 SSH 登录 Jetson 后的 Bash 中执行：

```bash
cd /home/ubuntu
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git checkout --detach 7f061f21f0a581ba234a1e233c9315b89d8e47d6
git submodule update --init --recursive
git rev-parse HEAD
```

为 HTTP server 安装最小 Python 依赖，不替换 ParkSight 已验证的 Jetson PyTorch：

```bash
/home/ubuntu/JetsonVLM/.venv-jetson/bin/python -m pip install \
  -r /home/ubuntu/TensorRT-Edge-LLM/requirements-server.txt
```

配置并以较低并行度编译，避免 8GB 板端因编译并发产生额外内存压力：

```bash
cd /home/ubuntu/TensorRT-Edge-LLM
mkdir -p build
cd build

PYBIND11_DIR=$(/home/ubuntu/JetsonVLM/.venv-jetson/bin/python \
  -m pybind11 --cmakedir)

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin \
  -DCUDA_CTK_VERSION=12.6 \
  -DENABLE_CUTE_DSL=ALL \
  -DBUILD_PYTHON_BINDINGS=ON \
  -Dpybind11_DIR="$PYBIND11_DIR"

cmake --build . --parallel 2
```

成功标准：至少存在以下文件和程序：

```bash
cd /home/ubuntu/TensorRT-Edge-LLM
test -f build/libNvInfer_edgellm_plugin.so
test -n "$(find build/pybind -maxdepth 1 -name '*_edgellm_runtime*.so' -print -quit)"
test -x build/examples/llm/llm_build
test -x build/examples/llm/llm_inference
test -x build/examples/multimodal/visual_build
```

## 6. Jetson：构建两个 engine

先检查 flow readiness：

```bash
cd /home/ubuntu/JetsonVLM
PYTHONPATH=src .venv-jetson/bin/python scripts/build_engine.py \
  --config configs/flows/build_qwen3_vl_2b_fp16_engines.json
```

只有输出中的 `ready` 为 `true` 才执行：

```bash
PYTHONPATH=src .venv-jetson/bin/python scripts/build_engine.py \
  --config configs/flows/build_qwen3_vl_2b_fp16_engines.json \
  --execute
```

入口会再次核对 Edge-LLM commit，依次执行 `llm_build` 和 `visual_build`，并拒绝覆盖
已有 engine。成功标准：

- `artifacts/engines/qwen3_vl_2b_fp16/llm/llm.engine`
- `artifacts/engines/qwen3_vl_2b_fp16/visual/visual.engine`
- LLM tokenizer/chat template/embedding 与 visual config/preprocessor sidecar
- `reports/flows/build_qwen3_vl_2b_fp16_engines.json` 的 `status` 为 `succeeded`

若发生 OOM，保存完整日志、`free -h`、`tegrastats` 和失败阶段；这仍是 FP16
实验结果，不删除后直接改称 INT4 成功。

## 7. Jetson：启动预构建 engine server

项目提供的入口只加载已有 engine，不隐式导出或重新构建：

```bash
cd /home/ubuntu/JetsonVLM
export EDGE_LLM_ROOT=/home/ubuntu/TensorRT-Edge-LLM
export PYTHONPATH=$EDGE_LLM_ROOT:$PWD/src
export EDGELLM_PLUGIN_PATH=$EDGE_LLM_ROOT/build/libNvInfer_edgellm_plugin.so
export LD_LIBRARY_PATH=$EDGE_LLM_ROOT/build:$LD_LIBRARY_PATH

.venv-jetson/bin/python scripts/serve_edgellm.py \
  --engine-root artifacts/engines/qwen3_vl_2b_fp16 \
  --host 127.0.0.1 \
  --port 8000
```

另开一个 Jetson SSH 终端验证：

```bash
curl --fail http://127.0.0.1:8000/health
```

成功标准：HTTP 连接成功并返回健康状态。server 是实验性接口，因此必须保留固定
Edge-LLM commit，不能把另一个版本的 server 与当前 engine 混用。

## 8. 单图与 PS2.0 Study

先选择 PS2.0 pilot 中的一张图片进行单图验收：

```bash
cd /home/ubuntu/JetsonVLM
PYTHONPATH=src .venv-jetson/bin/python -m parksight_vlm.app.analyze_image \
  --image data/raw/ps2.0/pilot/indoor/001.jpg \
  --runtime tensorrt_edge_llm_http \
  --backend-revision 7f061f21f0a581ba234a1e233c9315b89d8e47d6 \
  --model-revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --adapter-revision edge-http-v2 \
  --precision fp16 \
  --edge-url http://127.0.0.1:8000
```

实际图片路径以 manifest 的 `image_ref` 为准。成功标准：输出
`"succeeded": true`，并包含通过严格 schema 校验的 `assessment`。

单图成功后再运行完整 pilot：

```bash
PYTHONPATH=src .venv-jetson/bin/python -m parksight_vlm.app.run_study \
  --config configs/studies/jetson_edgellm_fp16_ps20_pilot.json
```

最终报告为 `reports/jetson_edgellm_fp16_ps20_pilot.json`。只有该报告与
`reports/jetson_transformers_fp16_ps20_pilot.json` 使用相同 workload identity、
样本和功耗模式时，才计算加速比和质量差异。
