"""把弱监督候选数据转换为可人工复核的 package、清单和离线页面输入。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL file must not be empty: {path}")
    return rows


def build_review_inputs(
    *, candidate_path: Path, output_directory: Path, label_source: str
) -> dict[str, Any]:
    candidates = _load_jsonl(candidate_path)
    package: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    risk_counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()

    for row in candidates:
        case_id = str(row.get("sample_id", "")).strip()
        image = Path(str(row.get("image", "")))
        assessment = row.get("assessment")
        if not case_id or not image.name or not isinstance(assessment, dict):
            raise ValueError(f"candidate row has invalid identity or assessment: {row}")
        if case_id in seen:
            raise ValueError(f"duplicate candidate case_id: {case_id}")
        seen.add(case_id)
        image_ref = image.name
        split = str(row.get("split", "unknown"))
        source_group_id = str(row.get("source_group_id", "")).strip()
        if not source_group_id:
            raise ValueError(f"candidate row has no source_group_id: {case_id}")
        manifest.append(
            {
                "case_id": case_id,
                "image_ref": image_ref,
                "source_group_id": source_group_id,
                "split": split,
            }
        )
        annotations.append({"case_id": case_id, "assessment": assessment})
        package_record = {
            "case_id": case_id,
            "image_ref": image_ref,
            "source_group_id": source_group_id,
            "split": split,
            "label_source": label_source,
            "review_status": "candidate",
            "candidate_assessment": assessment,
            "human_assessment": None,
            "review_note": "",
        }
        package.append(package_record)
        items.append(
            {
                "case_id": case_id,
                "image_ref": image_ref,
                "source_group_id": source_group_id,
                "split": split,
                "label_source": label_source,
                "candidate_assessment": assessment,
                "model_assessment": assessment,
                "failure": None,
                "raw_output": row.get("raw_output"),
                "risk_level_match": None,
                "event_false_positive": [],
                "event_false_negative": [],
                "review_priority": "high",
            }
        )
        risk_counts[str(assessment.get("risk_level"))] += 1
        event_counts.update(str(event) for event in assessment.get("events", []))

    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "ps86_extended_codex_candidate_v1.manifest.jsonl"
    annotations_path = output_directory / "ps86_extended_codex_candidate_v1.annotations.jsonl"
    package_path = output_directory / "ps86_extended_codex_review_package_v1.jsonl"
    error_review_path = output_directory / "ps86_extended_codex_error_review_v1.json"
    _write_jsonl(manifest_path, manifest)
    _write_jsonl(annotations_path, annotations)
    _write_jsonl(package_path, package)
    error_review = {
        "schema_version": "parksight_candidate_error_review_v1",
        "label_source": label_source,
        "manifest_path": str(manifest_path.resolve()),
        "annotations_path": str(annotations_path.resolve()),
        "summary": {
            "sample_count": len(items),
            "json_valid_count": len(items),
            "json_failure_count": 0,
            "review_priority_counts": {"high": len(items)},
            "risk_level_counts": dict(sorted(risk_counts.items())),
            "event_counts": dict(sorted(event_counts.items())),
            "weak_supervision_warning": "候选标签由基础模型生成，必须人工确认后才可进入正式训练",
        },
        "items": items,
    }
    error_review_path.write_text(
        json.dumps(error_review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "sample_count": len(items),
        "risk_level_counts": dict(sorted(risk_counts.items())),
        "event_counts": dict(sorted(event_counts.items())),
        "manifest": str(manifest_path),
        "annotations": str(annotations_path),
        "package": str(package_path),
        "error_review": str(error_review_path),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument(
        "--label-source",
        default="qwen3_vl_2b_base_weak_supervision_v1",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_review_inputs(
                candidate_path=args.candidate,
                output_directory=args.output_directory,
                label_source=args.label_source,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
