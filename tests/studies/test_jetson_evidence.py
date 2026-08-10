from __future__ import annotations

import unittest

from parksight_vlm.studies.jetson_evidence import (
    JetsonEvidenceError,
    parse_tegrastats,
    summarize_jetson_study,
)


class JetsonEvidenceTests(unittest.TestCase):
    def test_summary_keeps_backend_completion_separate_from_schema_validity(self) -> None:
        report = {
            "study_identity": {"study_id": "jetson-test"},
            "failure_summary": {"json_parse_error": 1},
            "records": [
                {
                    "assessment": None,
                    "raw_output": "{}",
                    "stage_timings": {"end_to_end_ms": 1000.0},
                    "output_tokens": 10,
                },
                {
                    "assessment": {"risk_level": "low"},
                    "raw_output": "{}",
                    "stage_timings": {"end_to_end_ms": 3000.0},
                    "output_tokens": 20,
                },
            ],
        }
        telemetry_lines = [
            "RAM 7000/7619MB SWAP 2500/12002MB GR3D_FREQ 99% "
            "gpu@60.5C VDD_IN 10000mW/9900mW"
        ]

        summary = summarize_jetson_study(
            study_report=report,
            tegrastats_lines=telemetry_lines,
        )

        runtime = summary["runtime_execution"]
        self.assertEqual(runtime["record_count"], 2)
        self.assertEqual(runtime["backend_completed_record_count"], 2)
        self.assertEqual(runtime["schema_valid_record_count"], 1)
        self.assertEqual(runtime["end_to_end_ms"]["p50"], 2000.0)
        self.assertEqual(
            runtime["output_tokens"][
                "aggregate_output_tokens_per_end_to_end_second"
            ],
            7.5,
        )
        self.assertEqual(summary["jetson_telemetry"]["vdd_in_w"]["maximum"], 10.0)

    def test_parse_tegrastats_rejects_log_without_complete_rows(self) -> None:
        with self.assertRaisesRegex(JetsonEvidenceError, "no complete telemetry"):
            parse_tegrastats(["not a tegrastats row"])


if __name__ == "__main__":
    unittest.main()
