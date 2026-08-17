import unittest

from scripts.quantize_qwen3_vl_int4_awq_domain import _iter_texts


class DomainQuantizationDataTests(unittest.TestCase):
    def test_loads_non_empty_text_rows(self) -> None:
        self.assertEqual(
            list(_iter_texts([{"text": "parking prompt"}])),
            ["parking prompt"],
        )

    def test_rejects_missing_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires non-empty text"):
            list(_iter_texts([{"sample_id": "missing-text"}]))


if __name__ == "__main__":
    unittest.main()
