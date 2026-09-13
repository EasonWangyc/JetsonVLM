"""汇总 Edge-LLM 低层 runtime 输出的阶段级 benchmark JSONL。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from parksight_vlm.tensorrt import (
    TensorRTValidationError,
    load_benchmark_samples,
    summarize_benchmark_samples,
    write_json,
)
from parksight_vlm.studies.jetson_evidence import JetsonEvidenceError, parse_tegrastats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        required=True,
        type=Path,
        help="Edge-LLM low-level runner 输出，每行一个 benchmark sample",
    )
    parser.add_argument(
        "--warmup-input-jsonl",
        type=Path,
        help="可选的 run_edgellm_benchmark warm-up JSONL，用于填充 cold_start_ms",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--metadata-json",
        type=Path,
        help="engine、设备、profile 和 runtime 配置的 JSON 快照",
    )
    parser.add_argument(
        "--provenance-json",
        type=Path,
        help="可选 engine provenance；会嵌入汇总 metadata 以绑定 engine SHA-256",
    )
    parser.add_argument(
        "--runtime-metadata-json",
        type=Path,
        help="可选服务启动身份；会嵌入汇总 metadata 以绑定 runtime 开关",
    )
    parser.add_argument(
        "--tegrastats",
        type=Path,
        help="可选的同次 Jetson tegrastats 原始日志",
    )
    args = parser.parse_args(argv)

    metadata: dict[str, Any] = {}
    if args.metadata_json is not None:
        metadata = _read_object(args.metadata_json, "benchmark metadata")
    if args.provenance_json is not None:
        provenance = _read_object(args.provenance_json, "engine provenance")
        if provenance.get("schema_version") != "parksight_tensorrt_engine_provenance_v1":
            raise TensorRTValidationError(
                "benchmark provenance has unsupported or missing schema_version"
            )
        if "engine" not in provenance or "configuration" not in provenance:
            raise TensorRTValidationError(
                "benchmark provenance must contain engine and configuration"
            )
        metadata["engine_provenance"] = provenance
    if args.runtime_metadata_json is not None:
        runtime_metadata = _read_object(
            args.runtime_metadata_json, "runtime metadata"
        )
        if runtime_metadata.get("schema_version") != "parksight_tensorrt_runtime_options_v1":
            raise TensorRTValidationError(
                "benchmark runtime metadata has unsupported or missing schema_version"
            )
        if "engines" not in runtime_metadata or "options" not in runtime_metadata:
            raise TensorRTValidationError(
                "benchmark runtime metadata must contain engines and options"
            )
        metadata["runtime_metadata"] = runtime_metadata

    samples = load_benchmark_samples(args.input_jsonl)
    warmup_samples = (
        load_benchmark_samples(args.warmup_input_jsonl)
        if args.warmup_input_jsonl is not None
        else None
    )
    summary = summarize_benchmark_samples(
        samples,
        metadata=metadata,
        warmup_samples=warmup_samples,
    )
    summary["evidence_sources"] = {"input_jsonl": str(args.input_jsonl)}
    if args.warmup_input_jsonl is not None:
        summary["evidence_sources"]["warmup_input_jsonl"] = str(
            args.warmup_input_jsonl
        )
    if args.metadata_json is not None:
        summary["evidence_sources"]["metadata_json"] = str(args.metadata_json)
    if args.provenance_json is not None:
        summary["evidence_sources"]["provenance_json"] = str(args.provenance_json)
    if args.runtime_metadata_json is not None:
        summary["evidence_sources"]["runtime_metadata_json"] = str(
            args.runtime_metadata_json
        )
    if args.tegrastats is not None:
        try:
            tegrastats_lines = args.tegrastats.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            summary["jetson_telemetry"] = parse_tegrastats(tegrastats_lines)
        except (OSError, JetsonEvidenceError) as error:
            raise TensorRTValidationError(
                f"cannot parse benchmark tegrastats: {args.tegrastats}"
            ) from error
        summary["evidence_sources"]["tegrastats"] = str(args.tegrastats)
    write_json(summary, args.output)
    print(args.output)
    return 0


def _read_object(path: Path, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TensorRTValidationError(f"cannot read {context}: {path}") from error
    if not isinstance(payload, dict):
        raise TensorRTValidationError(f"{context} must be an object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
