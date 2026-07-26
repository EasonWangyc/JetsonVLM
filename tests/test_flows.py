from __future__ import annotations

import unittest
from pathlib import Path

from parksight_vlm.flows import ExternalFlowPlan, FlowValidationError


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ExternalFlowPlanTests(unittest.TestCase):
    def test_loads_every_stage_template_with_stable_identity(self) -> None:
        expected_stages = {
            "train_lora.example.json": "train_lora",
            "merge_lora.example.json": "merge_lora",
            "export_model.example.json": "export_model",
            "build_engine.example.json": "build_engine",
        }
        for filename, expected_stage in expected_stages.items():
            with self.subTest(filename=filename):
                path = REPOSITORY_ROOT / "configs" / "flows" / filename
                first = ExternalFlowPlan.load(path)
                second = ExternalFlowPlan.load(path)
                self.assertEqual(first.stage, expected_stage)
                self.assertEqual(first.identity, second.identity)
                self.assertEqual(len(first.identity), 64)
                self.assertFalse(first.readiness_mapping()["ready"])

    def test_rejects_unknown_fields_and_unsafe_path_overlap(self) -> None:
        valid = {
            "flow_id": "flow",
            "stage": "train_lora",
            "working_directory": ".",
            "command": ["trainer"],
            "required_inputs": ["input.json"],
            "expected_outputs": ["output.json"],
            "record_path": "record.json",
            "log_path": "flow.log",
        }
        with self.assertRaises(FlowValidationError):
            ExternalFlowPlan.from_mapping(
                {**valid, "unexpected": True}, config_root=REPOSITORY_ROOT
            )
        with self.assertRaises(FlowValidationError):
            ExternalFlowPlan.from_mapping(
                {**valid, "record_path": "output.json"},
                config_root=REPOSITORY_ROOT,
            )


if __name__ == "__main__":
    unittest.main()
