"""从预构建的 LLM/visual engine 启动 Edge-LLM HTTP 服务。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def serve_prebuilt_engines(*, engine_root: Path, host: str, port: int) -> None:
    """加载预构建 engine，避免在 8GB Jetson 上隐式执行模型导出。"""
    try:
        from experimental.server import LLM
    except ImportError as error:
        raise RuntimeError(
            "无法导入 experimental.server；请把固定 commit 的 "
            "TensorRT Edge-LLM checkout 加入 PYTHONPATH，并构建 Python bindings"
        ) from error

    llm = LLM(
        engine_dir=str(engine_root / "llm"),
        visual_engine_dir=str(engine_root / "visual"),
    )
    llm.serve(host=host, port=port)


def configure_weight_streaming_budget(budget_bytes: int | None) -> None:
    """在导入 C++ runtime 前设置可选的 TensorRT 权重驻留预算。"""
    if budget_bytes is None:
        return
    if budget_bytes < 0:
        raise ValueError("weight streaming budget must not be negative")
    os.environ["EDGELLM_WEIGHT_STREAMING_BUDGET_BYTES"] = str(budget_bytes)


def configure_edge_llm_environment(
    edge_llm_root: Path | None,
    plugin_path: Path | None,
) -> None:
    """配置源码、pybind 和插件路径，避免依赖隐式 shell 环境。"""
    if edge_llm_root is not None:
        resolved_root = edge_llm_root.resolve()
        if not resolved_root.is_dir():
            raise FileNotFoundError(f"TensorRT Edge-LLM root not found: {resolved_root}")
        for import_path in (resolved_root, resolved_root / "build" / "pybind"):
            if import_path.is_dir() and str(import_path) not in sys.path:
                sys.path.insert(0, str(import_path))
        os.environ.setdefault("BUILD_DIR", str(resolved_root / "build"))
        if plugin_path is None and not os.environ.get("EDGELLM_PLUGIN_PATH"):
            discovered_plugin = resolved_root / "build" / "libNvInfer_edgellm_plugin.so"
            if discovered_plugin.is_file():
                os.environ["EDGELLM_PLUGIN_PATH"] = str(discovered_plugin)

    configured_plugin = plugin_path
    if configured_plugin is None and os.environ.get("EDGELLM_PLUGIN_PATH"):
        configured_plugin = Path(os.environ["EDGELLM_PLUGIN_PATH"])
    if configured_plugin is not None:
        resolved_plugin = configured_plugin.resolve()
        if not resolved_plugin.is_file():
            raise FileNotFoundError(f"Edge-LLM plugin not found: {resolved_plugin}")
        os.environ["EDGELLM_PLUGIN_PATH"] = str(resolved_plugin)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--weight-streaming-budget-bytes",
        type=int,
        default=None,
        help="TensorRT LLM 权重 GPU 驻留预算；8GB Jetson 可使用 0 以最大化节省",
    )
    parser.add_argument(
        "--edge-llm-root",
        type=Path,
        default=Path(os.environ["EDGE_LLM_ROOT"])
        if os.environ.get("EDGE_LLM_ROOT")
        else None,
        help="TensorRT Edge-LLM checkout；自动加入源码和 pybind import 路径",
    )
    parser.add_argument(
        "--plugin-path",
        type=Path,
        default=None,
        help="TensorRT Edge-LLM plugin .so；未指定时从 --edge-llm-root 自动发现",
    )
    args = parser.parse_args(argv)

    engine_root = Path(args.engine_root).resolve()
    required_engines = (
        engine_root / "llm" / "llm.engine",
        engine_root / "visual" / "visual.engine",
    )
    missing_engines = [
        str(path) for path in required_engines if not path.is_file()
    ]
    if missing_engines:
        raise FileNotFoundError(f"缺少预构建 engine：{missing_engines}")

    configure_edge_llm_environment(args.edge_llm_root, args.plugin_path)
    configure_weight_streaming_budget(args.weight_streaming_budget_bytes)
    serve_prebuilt_engines(
        engine_root=engine_root,
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
