"""从预构建的 LLM/visual engine 启动 Edge-LLM HTTP 服务。"""

from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
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

    serve_prebuilt_engines(
        engine_root=engine_root,
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
