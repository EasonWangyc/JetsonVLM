import unittest

from scripts.finetune_qwen3_vl_lora import (
    _oversample_non_low_records,
    _validate_label_provenance,
)


class LoraSamplingTests(unittest.TestCase):
    def test_oversamples_only_non_low_records(self) -> None:
        low = {"sample_id": "low", "assessment": {"risk_level": "low"}}
        medium = {
            "sample_id": "medium",
            "assessment": {"risk_level": "medium"},
        }
        result = _oversample_non_low_records([low, medium], factor=2)
        self.assertEqual(
            [record["sample_id"] for record in result],
            ["low", "medium", "medium"],
        )

    def test_rejects_invalid_factor(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be at least 1"):
            _oversample_non_low_records([], factor=0)

    def test_training_requires_human_confirmed_source(self) -> None:
        records = [{"label_source": "codex_visual_review_v1_single_pass"}]
        with self.assertRaisesRegex(ValueError, "candidate Codex labels are blocked"):
            _validate_label_provenance(
                records,
                {
                    "label_source": "codex_visual_review_v1_single_pass",
                    "allow_candidate_labels": False,
                },
            )

    def test_training_rejects_mixed_or_mismatched_sources(self) -> None:
        records = [
            {"label_source": "human_confirmed_v1"},
            {"label_source": "codex_visual_review_v1_single_pass"},
        ]
        with self.assertRaisesRegex(ValueError, "does not match"):
            _validate_label_provenance(
                records, {"label_source": "human_confirmed_v1"}
            )

    def test_training_accepts_human_confirmed_source(self) -> None:
        records = [{"label_source": "human_confirmed_v1"}]
        self.assertEqual(
            _validate_label_provenance(
                records, {"label_source": "human_confirmed_v1"}
            ),
            "human_confirmed_v1",
        )


if __name__ == "__main__":
    unittest.main()
