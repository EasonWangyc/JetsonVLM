"""使用固定 Qwen3-VL 基础模型生成可审计的泊车弱监督 LoRA 数据集。"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from parksight_vlm.assessment import ParkingAssessment
from parksight_vlm.inference import HuggingFaceQwen3VlBackend
from parksight_vlm.workload import FrozenWorkload


def source_group_id(image_path: Path) -> str:
    """从 PS2.0 文件名提取连续采集序列标识。"""
    return re.sub(r"[-_]\d+$", "", image_path.stem)


def select_group_disjoint_images(
    image_root: Path,
    *,
    train_count: int,
    validation_count: int,
    seed: int,
) -> list[tuple[Path, str, str]]:
    """每个来源组只取一张图，并按组划分 train/validation。"""
    grouped: dict[str, list[Path]] = defaultdict(list)
    for image_path in sorted(image_root.glob("*.jpg")):
        grouped[source_group_id(image_path)].append(image_path)

    required = train_count + validation_count
    if len(grouped) < required:
        raise ValueError(
            f"need {required} source groups, but only found {len(grouped)}"
        )

    rng = random.Random(seed)
    groups = sorted(grouped)
    rng.shuffle(groups)
    selected: list[tuple[Path, str, str]] = []
    for index, group_id in enumerate(groups[:required]):
        candidates = grouped[group_id]
        image_path = candidates[len(candidates) // 2]
        split = "train" if index < train_count else "validation"
        selected.append((image_path, group_id, split))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--validation-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    workload = FrozenWorkload.load(args.workload)
    selections = select_group_disjoint_images(
        args.image_root,
        train_count=args.train_count,
        validation_count=args.validation_count,
        seed=args.seed,
    )
    backend = HuggingFaceQwen3VlBackend(
        model_id=args.model,
        model_revision=args.model_revision,
        dtype="bfloat16",
        attn_implementation="sdpa",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    failures: list[dict[str, str]] = []
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, (image_path, group_id, split) in enumerate(selections, start=1):
            generation = backend.generate(image_path=image_path, workload=workload)
            try:
                assessment = ParkingAssessment.from_mapping(
                    json.loads(generation.raw_output)
                )
            except Exception as error:
                failures.append(
                    {
                        "image": image_path.name,
                        "error": f"{type(error).__name__}: {error}",
                        "raw_output": generation.raw_output,
                    }
                )
                print(f"[{index}/{len(selections)}] invalid {image_path.name}: {error}")
                continue

            record = {
                "sample_id": f"ps2-{image_path.stem}",
                "image": image_path.resolve().as_posix(),
                "source_group_id": group_id,
                "split": split,
                "label_source": "qwen3_vl_2b_base_weak_supervision",
                "model_revision": args.model_revision,
                "workload_identity": workload.identity,
                "assessment": assessment.to_mapping(),
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            handle.flush()
            written += 1
            print(f"[{index}/{len(selections)}] wrote {split} {image_path.name}")

    failure_path = args.output.with_suffix(".failures.json")
    failure_path.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected": len(selections),
                "written": written,
                "failed": len(failures),
                "output": str(args.output),
                "failures": str(failure_path),
            },
            ensure_ascii=False,
        )
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
