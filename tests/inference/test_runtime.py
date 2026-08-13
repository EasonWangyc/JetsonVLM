"""Runtime 执行记录与失败分类测试。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from parksight_vlm.assessment import ParkingAssessment
from parksight_vlm.casebook import DatasetSplit, ParkingCase
from parksight_vlm.inference import (
    EdgeLlmRuntime,
    EdgeLlmHttpBackend,
    ResourceSnapshot,
    RuntimeDependencyError,
    RuntimeFailureCategory,
    RuntimeGeneration,
    RuntimeRefusalError,
    StageTimings,
    TransformersRuntime,
)
from parksight_vlm.inference.transformers import (
    _require_cuda_architecture,
    build_qwen3_vl_chat_messages,
)
from parksight_vlm.workload import FrozenWorkload


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "inference"
WORKLOAD = FrozenWorkload.load(
    PROJECT_ROOT / "configs" / "workloads" / "parking_risk_v1.json"
)


class StaticBackend:
    def __init__(self, generation: RuntimeGeneration | Exception) -> None:
        self.generation = generation
        self.calls: list[Path] = []

    def generate(self, *, image_path: Path, workload: FrozenWorkload) -> RuntimeGeneration: # 与TransformersRuntime定义一致
        self.calls.append(image_path)
        if isinstance(self.generation, Exception):
            raise self.generation
        return self.generation


class RuntimeTests(unittest.TestCase):
    def test_qwen3_vl_messages_use_typed_content_items(self) -> None:
        image = object()

        messages = build_qwen3_vl_chat_messages(image=image, workload=WORKLOAD)

        self.assertEqual(
            messages[0],
            {
                "role": "system",
                "content": [{"type": "text", "text": WORKLOAD.system_prompt}],
            },
        )
        self.assertIs(messages[1]["content"][0]["image"], image)
        self.assertEqual(messages[1]["content"][1]["type"], "text")
        self.assertEqual(
            messages[1]["content"][1]["text"],
            WORKLOAD.render_user_prompt(),
        )

    def test_transformers_backend_rejects_incompatible_cuda_wheel(self) -> None:
        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def get_device_capability() -> tuple[int, int]:
                return (8, 7)

            @staticmethod
            def get_arch_list() -> list[str]:
                return ["sm_80", "sm_90"]

        class FakeTorch:
            __version__ = "2.9.1+cu126"
            cuda = FakeCuda()

            @staticmethod
            def ones(*args: object, **kwargs: object) -> object:
                raise RuntimeError("no compatible kernel image")

        with self.assertRaisesRegex(
            RuntimeDependencyError,
            "does not include CUDA kernels for sm_87",
        ):
            _require_cuda_architecture(FakeTorch())

    def test_transformers_runtime_records_validated_success(self) -> None:
        backend = StaticBackend(
            RuntimeGeneration(
                raw_output=json.dumps(self._assessment_payload()),
                stage_timings=StageTimings(preprocess_ms=2.0, decode_ms=5.0),
                resource_snapshot=ResourceSnapshot(peak_memory_mb=1024.0),
                output_tokens=42,
            )
        )
        runtime = TransformersRuntime(
            data_root=FIXTURE_ROOT,
            backend=backend,
            backend_revision="test",
            model_id="Qwen/Qwen3-VL-2B-Instruct",
            model_revision="test-revision",
        )

        record = runtime.analyze(self._case(), WORKLOAD)

        self.assertTrue(record.succeeded)
        self.assertEqual(record.assessment.risk_level.value, "medium")
        self.assertEqual(record.output_tokens, 42)
        self.assertEqual(record.resource_snapshot.peak_memory_mb, 1024.0)
        self.assertIsNotNone(record.stage_timings.end_to_end_ms)
        self.assertEqual(record.workload_identity, WORKLOAD.identity)
        self.assertEqual(backend.calls, [FIXTURE_ROOT / "scene.jpg"])
        self.assertEqual(record.to_mapping()["failure"], None)

    def test_edge_llm_runtime_preserves_invalid_json_failure(self) -> None:
        backend = StaticBackend(RuntimeGeneration(raw_output="not-json", output_tokens=2))
        runtime = EdgeLlmRuntime(
            data_root=FIXTURE_ROOT,
            backend=backend,
            backend_revision="test",
            model_id="Qwen/Qwen3-VL-2B-Instruct",
            model_revision="test-revision",
            adapter_revision="edge-test",
            precision="fp16",
        )

        record = runtime.analyze(self._case(), WORKLOAD)

        self.assertFalse(record.succeeded)
        self.assertEqual(record.failure.category, RuntimeFailureCategory.JSON_PARSE_ERROR)
        self.assertEqual(record.raw_output, "not-json")
        self.assertEqual(record.output_tokens, 2)

    def test_runtime_records_timeout_and_missing_input(self) -> None:
        timeout_runtime = TransformersRuntime(
            data_root=FIXTURE_ROOT,
            backend=StaticBackend(TimeoutError("generation timed out")),
            backend_revision="test",
            model_id="Qwen/Qwen3-VL-2B-Instruct",
            model_revision="test-revision",
        )
        timeout_record = timeout_runtime.analyze(self._case(), WORKLOAD)
        self.assertEqual(timeout_record.failure.category, RuntimeFailureCategory.TIMEOUT)

        missing_case = self._case(image_ref="missing.jpg")
        missing_record = timeout_runtime.analyze(missing_case, WORKLOAD)
        self.assertEqual(missing_record.failure.category, RuntimeFailureCategory.INPUT_ERROR)

    def test_runtime_records_explicit_model_refusal(self) -> None:
        runtime = TransformersRuntime(
            data_root=FIXTURE_ROOT,
            backend=StaticBackend(RuntimeRefusalError("request refused")),
            backend_revision="test",
            model_id="Qwen/Qwen3-VL-2B-Instruct",
            model_revision="test-revision",
        )

        record = runtime.analyze(self._case(), WORKLOAD)

        self.assertEqual(
            record.failure.category, RuntimeFailureCategory.MODEL_REFUSAL
        )

    def test_edge_llm_http_backend_uses_multimodal_chat_request(self) -> None:
        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "choices": [
                            {"message": {"content": json.dumps(self_payload)}}
                        ],
                        "usage": {"completion_tokens": 19},
                    }
                ).encode("utf-8")

        self_payload = self._assessment_payload()
        backend = EdgeLlmHttpBackend(base_url="http://127.0.0.1:8000")
        with patch(
            "parksight_vlm.inference.edge_llm.urlopen",
            return_value=FakeResponse(),
        ) as mocked_urlopen:
            generation = backend.generate(
                image_path=FIXTURE_ROOT / "scene.jpg",
                workload=WORKLOAD,
            )

        request = mocked_urlopen.call_args.args[0]
        request_payload = json.loads(request.data.decode("utf-8"))
        system_message = request_payload["messages"][0]
        image_item = request_payload["messages"][1]["content"][0]
        text_item = request_payload["messages"][1]["content"][1]
        self.assertEqual(
            system_message,
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": WORKLOAD.system_prompt}
                ],
            },
        )
        self.assertEqual(image_item["type"], "image")
        self.assertEqual(text_item["type"], "text")
        self.assertEqual(text_item["text"], WORKLOAD.render_user_prompt())
        self.assertEqual(generation.output_tokens, 19)
        self.assertEqual(json.loads(generation.raw_output), self_payload)

    @staticmethod
    def _assessment_payload() -> dict[str, object]:
        return {
            "schema_version": "parking_risk_v1",
            "risk_level": "medium",
            "events": ["narrow_passage"],
            "evidence": ["Vehicles leave a narrow maneuvering corridor."],
            "driver_advice": ["slow_down"],
        }

    @classmethod
    def _case(cls, image_ref: str = "scene.jpg") -> ParkingCase:
        return ParkingCase(
            case_id="case-001",
            image_ref=PurePosixPath(image_ref),
            source_group_id="sequence-a",
            split=DatasetSplit.TEST,
            reference_assessment=ParkingAssessment.from_mapping(cls._assessment_payload()),
        )


if __name__ == "__main__":
    unittest.main()
