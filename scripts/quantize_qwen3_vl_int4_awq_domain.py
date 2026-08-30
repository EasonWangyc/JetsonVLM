"""使用项目领域文本校准集量化 Qwen3-VL 的 LLM backbone。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("calibration dataset must not be empty")
    return records


def _iter_texts(records: list[dict[str, Any]]) -> Iterator[str]:
    for index, record in enumerate(records, start=1):
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"calibration row {index} requires non-empty text")
        yield text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-llm-root", required=True, type=Path)
    parser.add_argument("--expected-edge-llm-revision", required=True)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--num-samples", required=True, type=int)
    parser.add_argument(
        "--calibration-batch-size",
        type=int,
        default=1,
        help="calibration batch size; use 1 on 8GB Jetson devices",
    )
    parser.add_argument(
        "--logits-to-keep",
        type=int,
        default=1,
        help="limit calibration logits to reduce temporary GPU memory",
    )
    args = parser.parse_args()

    edge_llm_root = args.edge_llm_root.resolve()
    actual_revision = _git_revision(edge_llm_root)
    if actual_revision != args.expected_edge_llm_revision:
        raise RuntimeError(
            "TensorRT Edge-LLM revision mismatch: "
            f"expected {args.expected_edge_llm_revision}, got {actual_revision}"
        )
    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if args.calibration_batch_size <= 0:
        raise ValueError("calibration_batch_size must be positive")
    if args.logits_to_keep <= 0:
        raise ValueError("logits_to_keep must be positive")

    dataset_path = args.dataset_path.resolve()
    records = _load_records(dataset_path)
    if len(records) < args.num_samples:
        raise ValueError(
            f"calibration dataset has {len(records)} rows, fewer than "
            f"num_samples={args.num_samples}"
        )

    sys.path.insert(0, str(edge_llm_root))
    from tensorrt_edgellm.quantization import quantize as quantize_module
    from tensorrt_edgellm.quantization.quantize import quantize_and_export

    original_loader = quantize_module._text_calib_dataloader

    def low_memory_loader(
        tokenizer: Any,
        text_dataset: Any,
        *,
        batch_size: int = 1,
        num_samples: int = 512,
        max_length: int = 512,
    ) -> Any:
        return original_loader(
            tokenizer,
            text_dataset,
            batch_size=args.calibration_batch_size,
            num_samples=num_samples,
            max_length=max_length,
        )

    def low_memory_calibrate(model: Any, dataloader: Any) -> None:
        for data in dataloader:
            data = data.to(model.device)
            model(data, logits_to_keep=args.logits_to_keep)

    quantize_module._text_calib_dataloader = low_memory_loader
    quantize_module._calibrate = low_memory_calibrate

    selected_records = records[: args.num_samples]
    workload_identities = {
        record.get("workload_identity")
        for record in selected_records
    }
    if len(workload_identities) != 1 or None in workload_identities:
        raise ValueError(
            "selected calibration rows must share one non-empty workload_identity"
        )
    calibration_workload_identity = next(iter(workload_identities))

    def domain_text_dataset() -> Iterator[str]:
        yield from _iter_texts(selected_records)

    domain_text_dataset.calib_name = dataset_path.stem  # type: ignore[attr-defined]
    output_dir = args.output_dir.resolve()
    quantize_and_export(
        model_dir=str(args.model_dir.resolve()),
        output_dir=str(output_dir),
        quantization="int4_awq",
        dtype="fp16",
        device="cuda",
        text_dataset=domain_text_dataset,
        num_samples=args.num_samples,
    )

    provenance = {
        "status": "succeeded",
        "quantization": "int4_awq",
        "scope": "llm_backbone",
        "visual_precision": "fp16",
        "lm_head_precision": "fp16",
        "kv_cache_quantization": None,
        "model_dir": str(args.model_dir.resolve()),
        "calibration_dataset": str(dataset_path),
        "calibration_rows": args.num_samples,
        "calibration_batch_size": args.calibration_batch_size,
        "logits_to_keep": args.logits_to_keep,
        "calibration_sha256": _sha256(dataset_path),
        "calibration_workload_identity": calibration_workload_identity,
        "edge_llm_revision": actual_revision,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
