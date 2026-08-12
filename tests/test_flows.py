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

    def test_loads_pinned_qwen3_vl_int4_awq_flows(self) -> None:
        expected = {
            "quantize_qwen3_vl_2b_int4_awq.json": "quantize_model",
            "export_qwen3_vl_2b_int4_awq.json": "export_model",
            "build_qwen3_vl_2b_int4_awq_llm_engine_i768_k1024.json": (
                "build_engine"
            ),
        }

        plans = {
            filename: ExternalFlowPlan.load(
                REPOSITORY_ROOT / "configs" / "flows" / filename
            )
            for filename in expected
        }
        for filename, stage in expected.items():
            with self.subTest(filename=filename):
                self.assertEqual(plans[filename].stage, stage)
                self.assertNotIn("replace-with-", " ".join(plans[filename].command))

        quantize_command = plans[
            "quantize_qwen3_vl_2b_int4_awq.json"
        ].command
        self.assertIn("int4_awq", quantize_command)
        self.assertIn("128", quantize_command)
        self.assertNotIn("--visual_quantization", quantize_command)
        self.assertNotIn("--lm_head_quantization", quantize_command)

        build_command = plans[
            "build_qwen3_vl_2b_int4_awq_llm_engine_i768_k1024.json"
        ].command
        self.assertIn("768", build_command)
        self.assertIn("1024", build_command)
        self.assertNotIn("--enable-weight-streaming", build_command)

    def test_loads_pinned_qwen3_vl_fp16_deployment_flows(self) -> None:
        expected = {
            "export_qwen3_vl_2b_fp16.json": "export_model",
            "build_qwen3_vl_2b_fp16_engines.json": "build_engine",
            "build_qwen3_vl_2b_fp16_llm_engine.json": "build_engine",
            "build_qwen3_vl_2b_fp16_visual_engine.json": "build_engine",
        }

        for filename, stage in expected.items():
            with self.subTest(filename=filename):
                plan = ExternalFlowPlan.load(
                    REPOSITORY_ROOT / "configs" / "flows" / filename
                )
                serialized_command = " ".join(plan.command)
                self.assertEqual(plan.stage, stage)
                self.assertNotIn("replace-with-", serialized_command)
                self.assertIn("qwen3_vl_2b_fp16", serialized_command.lower())

        build_plan = ExternalFlowPlan.load(
            REPOSITORY_ROOT
            / "configs"
            / "flows"
            / "build_qwen3_vl_2b_fp16_engines.json"
        )
        self.assertIn(
            "7f061f21f0a581ba234a1e233c9315b89d8e47d6",
            build_plan.command,
        )
        export_plan = ExternalFlowPlan.load(
            REPOSITORY_ROOT
            / "configs"
            / "flows"
            / "export_qwen3_vl_2b_fp16.json"
        )
        export_outputs = {path.as_posix() for path in export_plan.expected_outputs}
        export_inputs = {path.as_posix() for path in export_plan.required_inputs}
        build_outputs = {path.as_posix() for path in build_plan.expected_outputs}
        self.assertTrue(
            any(path.endswith("/model.safetensors") for path in export_inputs)
        )
        self.assertFalse(
            any(path.endswith("/model.safetensors.index.json") for path in export_inputs)
        )
        self.assertTrue(
            any(path.endswith("/llm/tokenizer.json") for path in export_outputs)
        )
        self.assertTrue(
            any(path.endswith("/visual/config.json") for path in export_outputs)
        )
        self.assertTrue(
            any(path.endswith("/llm/tokenizer.json") for path in build_outputs)
        )
        self.assertTrue(
            any(path.endswith("/visual/config.json") for path in build_outputs)
        )

        llm_plan = ExternalFlowPlan.load(
            REPOSITORY_ROOT
            / "configs"
            / "flows"
            / "build_qwen3_vl_2b_fp16_llm_engine.json"
        )
        visual_plan = ExternalFlowPlan.load(
            REPOSITORY_ROOT
            / "configs"
            / "flows"
            / "build_qwen3_vl_2b_fp16_visual_engine.json"
        )
        self.assertIn("llm", llm_plan.command)
        self.assertIn("visual", visual_plan.command)
        self.assertTrue(
            all("/visual/" not in path.as_posix() for path in llm_plan.expected_outputs)
        )
        self.assertTrue(
            all("/llm/" not in path.as_posix() for path in visual_plan.expected_outputs)
        )

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
