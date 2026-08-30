"""复用弱监督生成失败文件中的原始输出，生成可审计的候选 JSONL。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from parksight_vlm.assessment import ParkingAssessment
from parksight_vlm.workload import FrozenWorkload

try:
    from scripts.generate_lora_dataset import normalize_json_fences, source_group_id
except ImportError:
    from generate_lora_dataset import normalize_json_fences, source_group_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-file", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replay-failures-output", type=Path)
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--validation-count", type=int, required=True)
    args = parser.parse_args()

    failures = json.loads(args.failure_file.read_text(encoding="utf-8"))
    required = args.train_count + args.validation_count
    if len(failures) != required:
        raise ValueError(f"failure file has {len(failures)} rows, expected {required}")
    workload = FrozenWorkload.load(args.workload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    replay_failures: list[dict[str, str]] = []
    normalized_count = 0
    for index, failure in enumerate(failures):
        image_name = str(failure["image"])
        image_path = args.image_root / image_name
        if not image_path.is_file():
            replay_failures.append({"image": image_name, "error": "image_not_found"})
            continue
        raw_output = str(failure.get("raw_output", ""))
        parse_output, normalized = normalize_json_fences(raw_output)
        normalization = "stripped_json_fence" if normalized else "none"
        normalized_count += int(normalized)
        try:
            assessment = ParkingAssessment.from_mapping(json.loads(parse_output))
        except Exception as error:
            replay_failures.append(
                {
                    "image": image_name,
                    "error": f"{type(error).__name__}: {error}",
                    "raw_output": raw_output,
                    "output_normalization": normalization,
                }
            )
            continue
        split = "train" if index < args.train_count else "validation"
        records.append(
            {
                "sample_id": f"ps2-{image_path.stem}",
                "image": image_path.resolve().as_posix(),
                "source_group_id": source_group_id(image_path),
                "split": split,
                "label_source": "qwen3_vl_2b_base_weak_supervision",
                "model_revision": args.model_revision,
                "workload_identity": workload.identity,
                "raw_output": raw_output,
                "output_normalization": normalization,
                "assessment": assessment.to_mapping(),
            }
        )

    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    replay_failure_path = args.replay_failures_output or args.output.with_suffix(
        ".replay_failures.json"
    )
    replay_failure_path.write_text(
        json.dumps(replay_failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected": len(failures),
                "written": len(records),
                "failed": len(replay_failures),
                "normalized_json_fences": normalized_count,
                "output": str(args.output),
                "failures": str(replay_failure_path),
            },
            ensure_ascii=False,
        )
    )
    return 0 if not replay_failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
