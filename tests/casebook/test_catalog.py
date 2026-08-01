"""JSONL 泊车样本目录加载与划分不变量测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

from parksight_vlm.casebook import (
    CasebookValidationError,
    DatasetSplit,
    ParkingCaseCatalog,
)


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "casebook"


class ParkingCaseCatalogTests(unittest.TestCase):
    def test_loads_matching_records_and_resolves_relative_image(self) -> None:
        catalog = self._load("valid_manifest.jsonl", "valid_annotations.jsonl")

        self.assertEqual(len(catalog.cases), 3)
        self.assertEqual(
            tuple(case.case_id for case in catalog.cases_in_split(DatasetSplit.TEST)),
            ("case-001", "case-002"),
        )
        image_path = catalog.cases[0].resolve_image(FIXTURE_ROOT)
        self.assertEqual(image_path, (FIXTURE_ROOT / "raw" / "case-001.jpg").resolve())

    def test_rejects_source_group_split_leakage(self) -> None:
        with self.assertRaisesRegex(
            CasebookValidationError, "source_group_id appears in multiple splits"
        ):
            self._load("leakage_manifest.jsonl", "leakage_annotations.jsonl")

    def test_rejects_duplicate_case_identifiers(self) -> None:
        with self.assertRaisesRegex(CasebookValidationError, "duplicate case_id"):
            self._load("duplicate_manifest.jsonl", "invalid_assessment_annotations.jsonl")

    def test_rejects_missing_or_extra_annotations(self) -> None:
        with self.assertRaisesRegex(
            CasebookValidationError,
            "missing annotations.*annotations without a manifest",
        ):
            self._load("unmatched_manifest.jsonl", "unmatched_annotations.jsonl")

    def test_rejects_invalid_image_reference_and_assessment(self) -> None:
        with self.assertRaisesRegex(CasebookValidationError, "stay below the data root"):
            self._load("invalid_image_manifest.jsonl", "single_case_001_annotations.jsonl")

        with self.assertRaisesRegex(CasebookValidationError, "invalid assessment"):
            self._load(
                "invalid_assessment_manifest.jsonl",
                "invalid_assessment_annotations.jsonl",
            )

    @staticmethod
    def _load(manifest_name: str, annotations_name: str) -> ParkingCaseCatalog:
        return ParkingCaseCatalog.load(
            FIXTURE_ROOT / manifest_name,
            FIXTURE_ROOT / annotations_name,
        )


if __name__ == "__main__":
    unittest.main()
