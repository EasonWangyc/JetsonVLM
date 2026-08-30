"""泊车评估跨字段语义审计测试。"""

from __future__ import annotations

import unittest

from parksight_vlm.assessment import (
    ParkingAssessment,
    SemanticIssue,
    audit_assessment_semantics,
)


class AssessmentSemanticAuditTests(unittest.TestCase):
    def test_low_risk_prepare_to_stop_is_flagged(self) -> None:
        assessment = self._assessment(
            risk_level="low",
            events=[],
            driver_advice=["prepare_to_stop"],
        )

        audit = audit_assessment_semantics(assessment)

        self.assertEqual(
            audit.issues,
            (SemanticIssue.LOW_RISK_WITH_PREPARE_TO_STOP,),
        )
        self.assertFalse(audit.is_consistent)

    def test_medium_prepare_to_stop_is_allowed(self) -> None:
        assessment = self._assessment(
            risk_level="medium",
            events=["vehicle_near_maneuver_path"],
            driver_advice=["slow_down", "prepare_to_stop"],
        )

        audit = audit_assessment_semantics(assessment)

        self.assertTrue(audit.is_consistent)

    def test_high_and_path_conflict_require_immediate_response(self) -> None:
        assessment = self._assessment(
            risk_level="high",
            events=["vru_near_maneuver_path"],
            driver_advice=["slow_down"],
        )

        audit = audit_assessment_semantics(assessment)

        self.assertEqual(
            audit.issues,
            (
                SemanticIssue.HIGH_RISK_WITHOUT_IMMEDIATE_RESPONSE,
                SemanticIssue.PATH_CONFLICT_WITHOUT_IMMEDIATE_RESPONSE,
            ),
        )

    def test_non_low_empty_events_are_reported(self) -> None:
        assessment = self._assessment(
            risk_level="medium",
            events=[],
            driver_advice=["maintain_observation"],
        )

        audit = audit_assessment_semantics(assessment)

        self.assertEqual(
            audit.issues,
            (SemanticIssue.NON_LOW_RISK_WITHOUT_EVENT,),
        )

    @staticmethod
    def _assessment(
        *, risk_level: str, events: list[str], driver_advice: list[str]
    ) -> ParkingAssessment:
        return ParkingAssessment.from_mapping(
            {
                "schema_version": "parking_risk_v1",
                "risk_level": risk_level,
                "events": events,
                "evidence": ["A visible scene cue supports the assessment."],
                "driver_advice": driver_advice,
            }
        )


if __name__ == "__main__":
    unittest.main()
