"""Unit tests for the parking assessment JSON contract."""

from __future__ import annotations

import unittest

from parksight_vlm.assessment import (
    ASSESSMENT_SCHEMA_VERSION,
    AssessmentValidationError,
    DriverAdvice,
    ParkingAssessment,
    ParkingRiskEvent,
    RiskLevel,
)


class ParkingAssessmentTests(unittest.TestCase):
    def test_from_mapping_parses_and_serializes_valid_payload(self) -> None:
        payload = {
            "schema_version": ASSESSMENT_SCHEMA_VERSION,
            "risk_level": "high",
            "events": [
                "vru_near_maneuver_path",
                "visibility_occlusion",
            ],
            "evidence": [
                "A pedestrian is beside the intended reversing path.",
                "A parked vehicle blocks part of the rearward view.",
            ],
            "driver_advice": ["slow_down", "prepare_to_stop"],
        }

        assessment = ParkingAssessment.from_mapping(payload)

        self.assertEqual(assessment.risk_level, RiskLevel.HIGH)
        self.assertEqual(
            assessment.events,
            (
                ParkingRiskEvent.VRU_NEAR_MANEUVER_PATH,
                ParkingRiskEvent.VISIBILITY_OCCLUSION,
            ),
        )
        self.assertEqual(
            assessment.driver_advice,
            (DriverAdvice.SLOW_DOWN, DriverAdvice.PREPARE_TO_STOP),
        )
        self.assertEqual(assessment.to_mapping(), payload)

    def test_low_risk_allows_no_risk_events(self) -> None:
        assessment = ParkingAssessment.from_mapping(
            {
                "schema_version": ASSESSMENT_SCHEMA_VERSION,
                "risk_level": "low",
                "events": [],
                "evidence": ["The maneuver path is visible and clear."],
                "driver_advice": ["maintain_observation"],
            }
        )

        self.assertEqual(assessment.events, ())

    def test_rejects_missing_and_unexpected_fields(self) -> None:
        payload = self._valid_payload()
        del payload["evidence"]
        payload["extra"] = "not permitted"

        with self.assertRaisesRegex(
            AssessmentValidationError,
            r"missing fields: \['evidence'\].*unexpected fields: \['extra'\]",
        ):
            ParkingAssessment.from_mapping(payload)

    def test_rejects_invalid_enum_values_and_sequences(self) -> None:
        for field_name, value in (
            ("risk_level", "critical"),
            ("events", ["unknown_event"]),
            ("driver_advice", ["drive_faster"]),
        ):
            with self.subTest(field_name=field_name):
                payload = self._valid_payload()
                payload[field_name] = value
                with self.assertRaises(AssessmentValidationError):
                    ParkingAssessment.from_mapping(payload)

    def test_rejects_non_mapping_and_non_array_fields(self) -> None:
        with self.assertRaisesRegex(AssessmentValidationError, "must be a mapping"):
            ParkingAssessment.from_mapping(object())

        for field_name in ("events", "evidence", "driver_advice"):
            with self.subTest(field_name=field_name):
                payload = self._valid_payload()
                payload[field_name] = "not an array"
                with self.assertRaisesRegex(AssessmentValidationError, "must be an array"):
                    ParkingAssessment.from_mapping(payload)

    def test_rejects_empty_or_duplicate_text_and_enum_entries(self) -> None:
        cases = (
            ("evidence", []),
            ("evidence", ["visible obstacle", "visible obstacle"]),
            ("evidence", ["   "]),
            ("events", ["narrow_passage", "narrow_passage"]),
            ("driver_advice", []),
            ("driver_advice", ["slow_down", "slow_down"]),
        )
        for field_name, value in cases:
            with self.subTest(field_name=field_name, value=value):
                payload = self._valid_payload()
                payload[field_name] = value
                with self.assertRaises(AssessmentValidationError):
                    ParkingAssessment.from_mapping(payload)

    @staticmethod
    def _valid_payload() -> dict[str, object]:
        return {
            "schema_version": ASSESSMENT_SCHEMA_VERSION,
            "risk_level": "medium",
            "events": ["narrow_passage"],
            "evidence": ["Vehicles leave a narrow maneuvering corridor."],
            "driver_advice": ["slow_down"],
        }


if __name__ == "__main__":
    unittest.main()
