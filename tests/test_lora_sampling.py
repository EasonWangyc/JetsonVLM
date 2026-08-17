import unittest

from scripts.finetune_qwen3_vl_lora import _oversample_non_low_records


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


if __name__ == "__main__":
    unittest.main()
