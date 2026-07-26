"""Parking-risk assessment data contract.

The model output and the human reference annotation share this representation.
It deliberately validates only the JSON contract; task-level correctness is
evaluated later by the studies module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar


ASSESSMENT_SCHEMA_VERSION = "parking_risk_v1"
EnumType = TypeVar("EnumType", bound=Enum)


class AssessmentValidationError(ValueError):
    """Raised when a payload does not conform to the assessment contract."""


class RiskLevel(str, Enum):
    """Overall parking risk level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ParkingRiskEvent(str, Enum):
    """Observable parking risk events in the frozen workload."""

    VRU_NEAR_MANEUVER_PATH = "vru_near_maneuver_path"
    VEHICLE_NEAR_MANEUVER_PATH = "vehicle_near_maneuver_path"
    FIXED_OBSTACLE_NEAR_PATH = "fixed_obstacle_near_path"
    NARROW_PASSAGE = "narrow_passage"
    VISIBILITY_OCCLUSION = "visibility_occlusion"
    PARKING_SPACE_CONFLICT = "parking_space_conflict"


class DriverAdvice(str, Enum):
    """Allowed parking-safety prompts for the driver."""

    MAINTAIN_OBSERVATION = "maintain_observation"
    SLOW_DOWN = "slow_down"
    YIELD = "yield"
    PREPARE_TO_STOP = "prepare_to_stop"
    CHANGE_MANEUVER_WHEN_SAFE = "change_maneuver_when_safe"


@dataclass(frozen=True, slots=True)
class ParkingAssessment:
    """A validated parking risk assessment that can be serialized as JSON."""

    schema_version: str
    risk_level: RiskLevel
    events: tuple[ParkingRiskEvent, ...]
    evidence: tuple[str, ...]
    driver_advice: tuple[DriverAdvice, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ParkingAssessment":
        """Parse and validate a strict JSON-compatible mapping."""
        if not isinstance(payload, Mapping):
            raise AssessmentValidationError("assessment payload must be a mapping")

        expected_fields = {
            "schema_version",
            "risk_level",
            "events",
            "evidence",
            "driver_advice",
        }
        actual_fields = set(payload)
        missing_fields = expected_fields - actual_fields
        unexpected_fields = actual_fields - expected_fields
        if missing_fields or unexpected_fields:
            details = []
            if missing_fields:
                details.append(f"missing fields: {sorted(missing_fields)}")
            if unexpected_fields:
                details.append(f"unexpected fields: {sorted(unexpected_fields)}")
            raise AssessmentValidationError("; ".join(details))

        schema_version = _parse_text(payload["schema_version"], "schema_version")
        if schema_version != ASSESSMENT_SCHEMA_VERSION:
            raise AssessmentValidationError(
                "unsupported schema_version: " f"{schema_version!r}"
            )

        risk_level = _parse_enum(payload["risk_level"], RiskLevel, "risk_level")
        events = _parse_enum_sequence(payload["events"], ParkingRiskEvent, "events")
        evidence = _parse_text_sequence(payload["evidence"], "evidence")
        driver_advice = _parse_enum_sequence(
            payload["driver_advice"], DriverAdvice, "driver_advice"
        )

        if not evidence:
            raise AssessmentValidationError("evidence must not be empty")
        if not driver_advice:
            raise AssessmentValidationError("driver_advice must not be empty")

        return cls(
            schema_version=schema_version,
            risk_level=risk_level,
            events=events,
            evidence=evidence,
            driver_advice=driver_advice,
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-compatible representation using the frozen field names."""
        return {
            "schema_version": self.schema_version,
            "risk_level": self.risk_level.value,
            "events": [event.value for event in self.events],
            "evidence": list(self.evidence),
            "driver_advice": [advice.value for advice in self.driver_advice],
        }


def _parse_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise AssessmentValidationError(f"{field_name} must be a string")
    normalized_value = value.strip()
    if not normalized_value:
        raise AssessmentValidationError(f"{field_name} must not be blank")
    return normalized_value


def _parse_text_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise AssessmentValidationError(f"{field_name} must be an array")
    values = tuple(_parse_text(item, field_name) for item in value)
    if len(set(values)) != len(values):
        raise AssessmentValidationError(f"{field_name} must not contain duplicates")
    return values


def _parse_enum(
    value: Any, enum_type: type[EnumType], field_name: str
) -> EnumType:
    if not isinstance(value, str):
        raise AssessmentValidationError(f"{field_name} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        allowed_values = ", ".join(member.value for member in enum_type)
        raise AssessmentValidationError(
            f"{field_name} must be one of: {allowed_values}"
        ) from error


def _parse_enum_sequence(
    value: Any,
    enum_type: type[EnumType],
    field_name: str,
) -> tuple[EnumType, ...]:
    if not isinstance(value, list):
        raise AssessmentValidationError(f"{field_name} must be an array")
    values: list[EnumType] = []
    for item in value:
        if not isinstance(item, str):
            raise AssessmentValidationError(f"{field_name} entries must be strings")
        try:
            values.append(enum_type(item))
        except ValueError as error:
            allowed_values = ", ".join(member.value for member in enum_type)
            raise AssessmentValidationError(
                f"{field_name} entries must be one of: {allowed_values}"
            ) from error
    if len(set(values)) != len(values):
        raise AssessmentValidationError(f"{field_name} must not contain duplicates")
    return tuple(values)
