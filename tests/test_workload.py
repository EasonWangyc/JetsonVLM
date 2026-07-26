"""Tests for frozen workload configuration and provenance."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from parksight_vlm.assessment import DriverAdvice, ParkingRiskEvent
from parksight_vlm.workload import FrozenWorkload, WorkloadValidationError


WORKLOAD_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "workloads"
    / "parking_risk_v1.json"
)


class FrozenWorkloadTests(unittest.TestCase):
    def test_loads_frozen_workload_and_builds_stable_identity(self) -> None:
        workload = FrozenWorkload.load(WORKLOAD_PATH)

        self.assertEqual(workload.workload_id, "parking_risk_v1")
        self.assertEqual((workload.input_size.width, workload.input_size.height), (448, 448))
        self.assertEqual(set(workload.risk_events), set(ParkingRiskEvent))
        self.assertEqual(set(workload.driver_advice), set(DriverAdvice))
        self.assertEqual(len(workload.fingerprint), 64)
        self.assertEqual(
            FrozenWorkload.from_mapping(workload.to_mapping()).identity,
            workload.identity,
        )

    def test_rejects_schema_drift_and_incomplete_event_set(self) -> None:
        payload = FrozenWorkload.load(WORKLOAD_PATH).to_mapping()
        payload["schema_version"] = "parking_risk_v2"
        with self.assertRaisesRegex(WorkloadValidationError, "unsupported schema_version"):
            FrozenWorkload.from_mapping(payload)

        payload = FrozenWorkload.load(WORKLOAD_PATH).to_mapping()
        payload["risk_events"] = payload["risk_events"][:-1]
        with self.assertRaisesRegex(WorkloadValidationError, "missing values"):
            FrozenWorkload.from_mapping(payload)

    def test_rejects_invalid_input_and_generation_parameters(self) -> None:
        valid_payload = FrozenWorkload.load(WORKLOAD_PATH).to_mapping()
        cases = (
            ("input_size", {"width": 0, "height": 448}),
            ("generation", {"max_new_tokens": True, "do_sample": False}),
            ("generation", {"max_new_tokens": 256, "do_sample": "false"}),
        )
        for field_name, invalid_value in cases:
            with self.subTest(field_name=field_name, invalid_value=invalid_value):
                payload = copy.deepcopy(valid_payload)
                payload[field_name] = invalid_value
                with self.assertRaises(WorkloadValidationError):
                    FrozenWorkload.from_mapping(payload)


if __name__ == "__main__":
    unittest.main()
