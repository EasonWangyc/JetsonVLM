from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_tensorrt_nsight_trace import summarize_trace


class NsightTraceSummaryTests(unittest.TestCase):
    def test_summarizes_kernel_categories_and_excludes_transfers_from_kernel_total(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace = Path(temporary_directory) / "trace.csv"
            with trace.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Duration (ns)", "Name"])
                writer.writeheader()
                writer.writerow({"Duration (ns)": "1000000", "Name": "gemm_w4a16_T2"})
                writer.writerow({"Duration (ns)": "2000000", "Name": "fmha_v2_flash_attention"})
                writer.writerow({"Duration (ns)": "3000000", "Name": "topKStage1"})
                writer.writerow({"Duration (ns)": "4000000", "Name": "[CUDA memcpy Device-to-Host]"})

            result = summarize_trace(trace, label="control")

        self.assertEqual(result["input"]["label"], "control")
        self.assertEqual(result["trace"]["event_count"], 4)
        self.assertEqual(result["trace"]["kernel_duration_ms"], 6.0)
        self.assertEqual(result["trace"]["transfer_duration_ms"], 4.0)
        self.assertEqual(result["categories"]["int4_w4a16"]["count"], 1)
        self.assertEqual(result["categories"]["attention_fmha"]["total_ms"], 2.0)
        self.assertEqual(result["categories"]["sampling"]["total_ms"], 3.0)
        self.assertEqual(result["categories"]["memcpy"]["count"], 1)
        self.assertEqual(result["top_kernels"][0]["name"], "topKStage1")

    def test_rejects_missing_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace = Path(temporary_directory) / "trace.csv"
            trace.write_text("Name\nfoo\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                summarize_trace(trace)
