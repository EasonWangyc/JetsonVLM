"""将人工确认后的 review package 转换为标准 assessment annotation JSONL。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from parksight_vlm.assessment import ParkingAssessment


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not records:
        raise ValueError(f"JSONL file must not be empty: {path}")
    return records


def finalize_review_package(
    *,
    package_path: Path,
    annotations_output: Path,
    summary_output: Path | None = None,
) -> dict[str, Any]:
    package = _load_jsonl(package_path)
    annotations: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    for record in package:
        required = {
            "case_id",
            "image_ref",
            "source_group_id",
            "split",
            "label_source",
            "review_status",
            "candidate_assessment",
            "human_assessment",
            "review_note",
        }
        if set(record) != required:
            raise ValueError("review package has invalid fields")
        case_id = str(record["case_id"])
        status = str(record["review_status"])
        statuses[status] += 1
        if status not in {"confirmed", "corrected"}:
            raise ValueError(
                f"review record is not finalized: {case_id} has status {status!r}"
            )
        if record["human_assessment"] is None:
            raise ValueError(f"missing human assessment: {case_id}")
        assessment = ParkingAssessment.from_mapping(record["human_assessment"])
        annotations.append({"case_id": case_id, "assessment": assessment.to_mapping()})

    if len({record["case_id"] for record in annotations}) != len(annotations):
        raise ValueError("review package contains duplicate case_id values")

    annotations_output.parent.mkdir(parents=True, exist_ok=True)
    with annotations_output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in annotations:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")

    summary = {
        "package_id": "ps80_codex_review_package_v1",
        "source_package": str(package_path),
        "sample_count": len(annotations),
        "review_status_counts": dict(sorted(statuses.items())),
        "annotations_output": str(annotations_output),
    }
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--annotations-output", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            finalize_review_package(
                package_path=args.package,
                annotations_output=args.annotations_output,
                summary_output=args.summary_output,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
