"""Codex 候选标注复盘工作流测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_label_review_sheets import _resolve_image
from scripts.build_review_package import build_review_package
from scripts.finalize_review_package import finalize_review_package


class ReviewWorkflowTests(unittest.TestCase):
    def test_resolve_image_supports_nested_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nested = root / "sequence" / "frame.jpg"
            nested.parent.mkdir(parents=True)
            nested.write_bytes(b"image")
            resolved = _resolve_image(
                {"case_id": "sample-1", "image_ref": "processed/frame.jpg"},
                root,
            )
            self.assertEqual(resolved, nested)

    def test_build_review_package_preserves_candidate_and_human_fields(self) -> None:
        assessment = {
            "schema_version": "parking_risk_v1",
            "risk_level": "low",
            "events": [],
            "evidence": ["可见区域内未发现风险目标。"],
            "driver_advice": ["maintain_observation"],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = root / "manifest.jsonl"
            annotations = root / "annotations.jsonl"
            output = root / "review-package.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "case_id": "sample-1",
                        "image_ref": "raw/sample.jpg",
                        "source_group_id": "group-1",
                        "split": "train",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            annotations.write_text(
                json.dumps(
                    {"case_id": "sample-1", "assessment": assessment},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            summary = build_review_package(
                manifest_path=manifest,
                candidate_annotations_path=annotations,
                output_path=output,
            )
            record = json.loads(output.read_text(encoding="utf-8").strip())

        self.assertEqual(summary["sample_count"], 1)
        self.assertEqual(record["review_status"], "candidate")
        self.assertEqual(record["candidate_assessment"], assessment)
        self.assertIsNone(record["human_assessment"])
        self.assertEqual(record["review_note"], "")

    def test_finalize_review_package_requires_confirmed_human_assessment(self) -> None:
        assessment = {
            "schema_version": "parking_risk_v1",
            "risk_level": "low",
            "events": [],
            "evidence": ["可见区域内未发现风险目标。"],
            "driver_advice": ["maintain_observation"],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            package = root / "package.jsonl"
            output = root / "annotations.jsonl"
            package.write_text(
                json.dumps(
                    {
                        "case_id": "sample-1",
                        "image_ref": "raw/sample.jpg",
                        "source_group_id": "group-1",
                        "split": "train",
                        "label_source": "codex_visual_review_v1_single_pass",
                        "review_status": "confirmed",
                        "candidate_assessment": assessment,
                        "human_assessment": assessment,
                        "review_note": "已确认",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            summary = finalize_review_package(
                package_path=package,
                annotations_output=output,
            )
            record = json.loads(output.read_text(encoding="utf-8").strip())

        self.assertEqual(summary["sample_count"], 1)
        self.assertEqual(summary["review_status_counts"], {"confirmed": 1})
        self.assertEqual(record, {"case_id": "sample-1", "assessment": assessment})


if __name__ == "__main__":
    unittest.main()
