"""Codex 候选标注复盘工作流测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_label_review_sheets import _resolve_image
from scripts.build_review_package import build_review_package
from scripts.apply_review_decisions import apply_review_decisions, create_decision_template
from scripts.finalize_review_package import finalize_review_package
from scripts.inspect_review_package import inspect_review_package


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
        self.assertEqual(
            summary["candidate_human_comparison"]["event_micro_f1"],
            0.0,
        )
        self.assertEqual(
            summary["candidate_human_comparison"]["risk_level_accuracy"],
            1.0,
        )
        self.assertEqual(record, {"case_id": "sample-1", "assessment": assessment})

    def test_finalize_rejects_corrected_review_without_note(self) -> None:
        candidate = {
            "schema_version": "parking_risk_v1",
            "risk_level": "low",
            "events": [],
            "evidence": ["可见区域内未发现风险目标。"],
            "driver_advice": ["maintain_observation"],
        }
        corrected = {**candidate, "risk_level": "medium"}
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            package = root / "package.jsonl"
            package.write_text(
                json.dumps(
                    {
                        "case_id": "sample-1",
                        "image_ref": "raw/sample.jpg",
                        "source_group_id": "group-1",
                        "split": "train",
                        "label_source": "codex_visual_review_v1_single_pass",
                        "review_status": "corrected",
                        "candidate_assessment": candidate,
                        "human_assessment": corrected,
                        "review_note": "",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "requires review_note"):
                finalize_review_package(
                    package_path=package,
                    annotations_output=root / "annotations.jsonl",
                )

    def test_finalize_reports_corrected_candidate_agreement_metrics(self) -> None:
        candidate = {
            "schema_version": "parking_risk_v1",
            "risk_level": "low",
            "events": [],
            "evidence": ["未见近距离风险目标。"],
            "driver_advice": ["maintain_observation"],
        }
        human = {
            "schema_version": "parking_risk_v1",
            "risk_level": "medium",
            "events": ["narrow_passage"],
            "evidence": ["通行空间较窄，需要减速观察。"],
            "driver_advice": ["slow_down"],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            package = root / "package.jsonl"
            package.write_text(
                json.dumps(
                    {
                        "case_id": "sample-1",
                        "image_ref": "raw/sample.jpg",
                        "source_group_id": "group-1",
                        "split": "train",
                        "label_source": "codex_visual_review_v1_single_pass",
                        "review_status": "corrected",
                        "candidate_assessment": candidate,
                        "human_assessment": human,
                        "review_note": "候选风险等级和事件均需修正",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            summary = finalize_review_package(
                package_path=package,
                annotations_output=root / "annotations.jsonl",
            )

        comparison = summary["candidate_human_comparison"]
        self.assertEqual(comparison["risk_level_accuracy"], 0.0)
        self.assertEqual(comparison["event_micro_precision"], 0.0)
        self.assertEqual(comparison["event_micro_recall"], 0.0)
        self.assertEqual(comparison["event_micro_f1"], 0.0)
        self.assertEqual(comparison["assessment_changed"], 1)
        self.assertEqual(comparison["risk_levels_changed"], 1)
        self.assertEqual(comparison["event_sets_changed"], 1)

    def test_inspect_reports_incomplete_package_without_writing(self) -> None:
        assessment = {
            "schema_version": "parking_risk_v1",
            "risk_level": "low",
            "events": [],
            "evidence": ["未见近距离风险目标。"],
            "driver_advice": ["maintain_observation"],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            package = Path(temporary_directory) / "package.jsonl"
            package.write_text(
                json.dumps(
                    {
                        "case_id": "sample-1",
                        "image_ref": "raw/sample.jpg",
                        "source_group_id": "group-1",
                        "split": "train",
                        "label_source": "codex_visual_review_v1_single_pass",
                        "review_status": "candidate",
                        "candidate_assessment": assessment,
                        "human_assessment": None,
                        "review_note": "",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            report = inspect_review_package(package)

        self.assertFalse(report["ready_for_finalize"])
        self.assertEqual(report["pending_count"], 1)
        self.assertEqual(report["pending_case_ids"], ["sample-1"])
        self.assertEqual(report["candidate_risk_level_counts"], {"low": 1})

    def test_apply_review_decisions_supports_incremental_confirmation(self) -> None:
        assessment = {
            "schema_version": "parking_risk_v1",
            "risk_level": "low",
            "events": [],
            "evidence": ["未见近距离风险目标。"],
            "driver_advice": ["maintain_observation"],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            package = root / "package.jsonl"
            decisions = root / "decisions.jsonl"
            output = root / "updated-package.jsonl"
            package.write_text(
                json.dumps(
                    {
                        "case_id": "sample-1",
                        "image_ref": "raw/sample.jpg",
                        "source_group_id": "group-1",
                        "split": "train",
                        "label_source": "codex_visual_review_v1_single_pass",
                        "review_status": "candidate",
                        "candidate_assessment": assessment,
                        "human_assessment": None,
                        "review_note": "",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            decisions.write_text(
                json.dumps(
                    {
                        "case_id": "sample-1",
                        "review_status": "confirmed",
                        "human_assessment": None,
                        "review_note": "人工确认候选结果",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            summary = apply_review_decisions(
                package_path=package,
                decisions_path=decisions,
                output_path=output,
                require_complete=True,
            )
            record = json.loads(output.read_text(encoding="utf-8").strip())

        self.assertEqual(summary["finalized_count"], 1)
        self.assertTrue(summary["ready_for_finalize"])
        self.assertEqual(record["review_status"], "confirmed")
        self.assertEqual(record["human_assessment"], assessment)

    def test_apply_review_decisions_preserves_pending_records(self) -> None:
        assessment = {
            "schema_version": "parking_risk_v1",
            "risk_level": "low",
            "events": [],
            "evidence": ["未见近距离风险目标。"],
            "driver_advice": ["maintain_observation"],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            package = root / "package.jsonl"
            decisions = root / "decisions.jsonl"
            output = root / "updated-package.jsonl"
            records = []
            for case_id in ("sample-1", "sample-2"):
                records.append(
                    {
                        "case_id": case_id,
                        "image_ref": f"raw/{case_id}.jpg",
                        "source_group_id": case_id,
                        "split": "train",
                        "label_source": "codex_visual_review_v1_single_pass",
                        "review_status": "candidate",
                        "candidate_assessment": assessment,
                        "human_assessment": None,
                        "review_note": "",
                    }
                )
            package.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
                encoding="utf-8",
            )
            decisions.write_text(
                json.dumps(
                    {
                        "case_id": "sample-1",
                        "review_status": "confirmed",
                        "human_assessment": None,
                        "review_note": "人工确认",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            summary = apply_review_decisions(
                package_path=package,
                decisions_path=decisions,
                output_path=output,
            )
            updated = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(summary["decision_count"], 1)
        self.assertEqual(summary["pending_count"], 1)
        self.assertEqual(updated[0]["review_status"], "confirmed")
        self.assertEqual(updated[1]["review_status"], "candidate")

    def test_apply_review_decisions_requires_note_for_correction(self) -> None:
        candidate = {
            "schema_version": "parking_risk_v1",
            "risk_level": "low",
            "events": [],
            "evidence": ["未见近距离风险目标。"],
            "driver_advice": ["maintain_observation"],
        }
        corrected = {**candidate, "risk_level": "medium"}
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            package = root / "package.jsonl"
            decisions = root / "decisions.jsonl"
            package.write_text(
                json.dumps(
                    {
                        "case_id": "sample-1",
                        "image_ref": "raw/sample.jpg",
                        "source_group_id": "group-1",
                        "split": "train",
                        "label_source": "codex_visual_review_v1_single_pass",
                        "review_status": "candidate",
                        "candidate_assessment": candidate,
                        "human_assessment": None,
                        "review_note": "",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            decisions.write_text(
                json.dumps(
                    {
                        "case_id": "sample-1",
                        "review_status": "corrected",
                        "human_assessment": corrected,
                        "review_note": "",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "requires review_note"):
                apply_review_decisions(
                    package_path=package,
                    decisions_path=decisions,
                    output_path=root / "updated.jsonl",
                )

    def test_create_decision_template_is_not_ready_for_apply(self) -> None:
        assessment = {
            "schema_version": "parking_risk_v1",
            "risk_level": "low",
            "events": [],
            "evidence": ["未见近距离风险目标。"],
            "driver_advice": ["maintain_observation"],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            package = root / "package.jsonl"
            template = root / "decisions-template.jsonl"
            package.write_text(
                json.dumps(
                    {
                        "case_id": "sample-1",
                        "image_ref": "raw/sample.jpg",
                        "source_group_id": "group-1",
                        "split": "train",
                        "label_source": "codex_visual_review_v1_single_pass",
                        "review_status": "candidate",
                        "candidate_assessment": assessment,
                        "human_assessment": None,
                        "review_note": "",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            summary = create_decision_template(
                package_path=package,
                output_path=template,
            )
            record = json.loads(template.read_text(encoding="utf-8").strip())

        self.assertEqual(summary["sample_count"], 1)
        self.assertFalse(summary["ready_for_apply"])
        self.assertEqual(record["review_status"], "candidate")
        self.assertIsNone(record["human_assessment"])


if __name__ == "__main__":
    unittest.main()
