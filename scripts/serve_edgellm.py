"""从预构建的 LLM/visual engine 启动 Edge-LLM HTTP 服务。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _resolve_engine_directories(
    *,
    engine_root: Path | None,
    llm_engine_root: Path | None,
    visual_engine_root: Path | None,
) -> tuple[Path, Path]:
    if engine_root is not None:
        if llm_engine_root is not None or visual_engine_root is not None:
            raise ValueError(
                "use either --engine-root or both --llm-engine-root and "
                "--visual-engine-root"
            )
        return engine_root / "llm", engine_root / "visual"
    if llm_engine_root is None or visual_engine_root is None:
        raise ValueError(
            "provide --engine-root or both --llm-engine-root and "
            "--visual-engine-root"
        )
    return llm_engine_root, visual_engine_root


def serve_prebuilt_engines(
    *,
    engine_root: Path | None = None,
    llm_engine_root: Path | None = None,
    visual_engine_root: Path | None = None,
    host: str,
    port: int,
) -> None:
    """加载预构建 engine，避免在 8GB Jetson 上隐式执行模型导出。"""
    try:
        from experimental.server import LLM
    except ImportError as error:
        raise RuntimeError(
            "无法导入 experimental.server；请把固定 commit 的 "
            "TensorRT Edge-LLM checkout 加入 PYTHONPATH，并构建 Python bindings"
        ) from error

    llm_root, visual_root = _resolve_engine_directories(
        engine_root=engine_root,
        llm_engine_root=llm_engine_root,
        visual_engine_root=visual_engine_root,
    )
    llm = LLM(engine_dir=str(llm_root), visual_engine_dir=str(visual_root))
    llm.serve(host=host, port=port)


def deployment_readiness(
    *,
    engine_root: Path | None = None,
    llm_engine_root: Path | None = None,
    visual_engine_root: Path | None = None,
    edge_llm_root: Path | None,
    plugin_path: Path | None,
) -> dict[str, object]:
    """Check static deployment inputs without importing or loading the runtime."""
    try:
        llm_root, visual_root = _resolve_engine_directories(
            engine_root=engine_root,
            llm_engine_root=llm_engine_root,
            visual_engine_root=visual_engine_root,
        )
    except ValueError as error:
        return {"ready": False, "configuration_error": str(error)}
    llm_root = llm_root.resolve()
    visual_root = visual_root.resolve()
    required_engines = (llm_root / "llm.engine", visual_root / "visual.engine")
    missing_engines = [
        str(path) for path in required_engines if not path.is_file()
    ]
    report: dict[str, object] = {
        "llm_engine_root": str(llm_root),
        "visual_engine_root": str(visual_root),
        "required_engines": [str(path) for path in required_engines],
        "missing_engines": missing_engines,
        "ready": not missing_engines,
    }
    try:
        configure_edge_llm_environment(edge_llm_root, plugin_path)
    except (FileNotFoundError, ValueError) as error:
        report["ready"] = False
        report["configuration_error"] = str(error)
    if edge_llm_root is not None:
        report["edge_llm_root"] = str(edge_llm_root.resolve())
    configured_plugin = os.environ.get("EDGELLM_PLUGIN_PATH")
    if configured_plugin:
        report["plugin_path"] = configured_plugin
    return report


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
    parser.add_argument("--engine-root")
    parser.add_argument(
        "--llm-engine-root",
        type=Path,
        help="包含 llm.engine 的目录；与 --visual-engine-root 配对使用",
    )
    parser.add_argument(
        "--visual-engine-root",
        type=Path,
        help="包含 visual.engine 的目录；与 --llm-engine-root 配对使用",
    )
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
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查 engine、源码和 plugin 路径，不导入或加载 GPU runtime",
    )
    args = parser.parse_args(argv)

    engine_root = Path(args.engine_root).resolve() if args.engine_root else None
    if args.check_only:
        report = deployment_readiness(
            engine_root=engine_root,
            llm_engine_root=args.llm_engine_root,
            visual_engine_root=args.visual_engine_root,
            edge_llm_root=args.edge_llm_root,
            plugin_path=args.plugin_path,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ready"] else 2

    try:
        llm_engine_root, visual_engine_root = _resolve_engine_directories(
            engine_root=engine_root,
            llm_engine_root=args.llm_engine_root,
            visual_engine_root=args.visual_engine_root,
        )
    except ValueError as error:
        parser.error(str(error))
    llm_engine_root = llm_engine_root.resolve()
    visual_engine_root = visual_engine_root.resolve()
    required_engines = (
        llm_engine_root / "llm.engine",
        visual_engine_root / "visual.engine",
    )
    missing_engines = [
        str(path) for path in required_engines if not path.is_file()
    ]
    if missing_engines:
        raise FileNotFoundError(f"缺少预构建 engine：{missing_engines}")

    configure_edge_llm_environment(args.edge_llm_root, args.plugin_path)
    configure_weight_streaming_budget(args.weight_streaming_budget_bytes)
    serve_prebuilt_engines(
        llm_engine_root=llm_engine_root,
        visual_engine_root=visual_engine_root,
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
