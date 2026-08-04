"""使用固定版本的 TensorRT Edge-LLM 构建 VLM 的两个 engine。"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def build_commands(
    *,
    edge_root: Path,
    onnx_root: Path,
    engine_root: Path,
    max_batch_size: int,
    max_input_len: int,
    max_kv_cache_capacity: int,
    min_image_tokens: int,
    max_image_tokens: int,
    max_image_tokens_per_image: int,
) -> tuple[list[str], list[str]]:
    """生成先 LLM、后视觉编码器的两条确定性构建命令。"""
    llm_command = [
        str(edge_root / "build" / "examples" / "llm" / "llm_build"),
        "--onnxDir",
        str(onnx_root / "llm"),
        "--engineDir",
        str(engine_root / "llm"),
        "--maxBatchSize",
        str(max_batch_size),
        "--maxInputLen",
        str(max_input_len),
        "--maxKVCacheCapacity",
        str(max_kv_cache_capacity),
    ]
    visual_command = [
        str(
            edge_root
            / "build"
            / "examples"
            / "multimodal"
            / "visual_build"
        ),
        "--onnxDir",
        str(onnx_root / "visual"),
        "--engineDir",
        str(engine_root),
        "--minImageTokens",
        str(min_image_tokens),
        "--maxImageTokens",
        str(max_image_tokens),
        "--maxImageTokensPerImage",
        str(max_image_tokens_per_image),
    ]
    return llm_command, visual_command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-llm-root", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--onnx-root", required=True)
    parser.add_argument("--engine-root", required=True)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-kv-cache-capacity", type=int, default=2048)
    parser.add_argument("--min-image-tokens", type=int, default=8)
    parser.add_argument("--max-image-tokens", type=int, default=2048)
    parser.add_argument("--max-image-tokens-per-image", type=int, default=2048)
    args = parser.parse_args(argv)

    edge_root = Path(args.edge_llm_root).resolve()
    onnx_root = Path(args.onnx_root).resolve()
    engine_root = Path(args.engine_root).resolve()
    llm_builder = edge_root / "build" / "examples" / "llm" / "llm_build"
    visual_builder = (
        edge_root
        / "build"
        / "examples"
        / "multimodal"
        / "visual_build"
    )
    required_paths = (
        llm_builder,
        visual_builder,
        onnx_root / "llm" / "model.onnx",
        onnx_root / "visual" / "model.onnx",
    )
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"缺少 engine 构建输入：{missing_paths}")

    revision_result = subprocess.run(
        ["git", "-C", str(edge_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual_revision = revision_result.stdout.strip()
    if actual_revision != args.expected_revision:
        raise RuntimeError(
            "TensorRT Edge-LLM revision 不匹配："
            f"expected={args.expected_revision}, actual={actual_revision}"
        )

    llm_engine_dir = engine_root / "llm"
    visual_engine_dir = engine_root / "visual"
    expected_outputs = (
        llm_engine_dir / "llm.engine",
        visual_engine_dir / "visual.engine",
    )
    existing_outputs = [str(path) for path in expected_outputs if path.exists()]
    if existing_outputs:
        raise FileExistsError(f"拒绝覆盖已有 engine：{existing_outputs}")
    llm_engine_dir.mkdir(parents=True, exist_ok=True)
    visual_engine_dir.mkdir(parents=True, exist_ok=True)

    llm_command, visual_command = build_commands(
        edge_root=edge_root,
        onnx_root=onnx_root,
        engine_root=engine_root,
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        max_kv_cache_capacity=args.max_kv_cache_capacity,
        min_image_tokens=args.min_image_tokens,
        max_image_tokens=args.max_image_tokens,
        max_image_tokens_per_image=args.max_image_tokens_per_image,
    )
    subprocess.run(llm_command, check=True)
    subprocess.run(visual_command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
