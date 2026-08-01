"""泊车风险评估数据契约。

模型输出与人工参考标注共用此表示。这里有意只校验 JSON 契约；任务层面的正确性
由 studies 模块在后续评估。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar


ASSESSMENT_SCHEMA_VERSION = "parking_risk_v1"
EnumType = TypeVar("EnumType", bound=Enum)


class AssessmentValidationError(ValueError):
    """当载荷不符合评估契约时抛出。"""


class RiskLevel(str, Enum):
    """泊车场景的总体风险等级。"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ParkingRiskEvent(str, Enum):
    """冻结 workload 中可观察的泊车风险事件。"""

    VRU_NEAR_MANEUVER_PATH = "vru_near_maneuver_path"
    VEHICLE_NEAR_MANEUVER_PATH = "vehicle_near_maneuver_path"
    FIXED_OBSTACLE_NEAR_PATH = "fixed_obstacle_near_path"
    NARROW_PASSAGE = "narrow_passage"
    VISIBILITY_OCCLUSION = "visibility_occlusion"
    PARKING_SPACE_CONFLICT = "parking_space_conflict"


class DriverAdvice(str, Enum):
    """允许提供给驾驶员的泊车安全提示。"""

    MAINTAIN_OBSERVATION = "maintain_observation"
    SLOW_DOWN = "slow_down"
    YIELD = "yield"
    PREPARE_TO_STOP = "prepare_to_stop"
    CHANGE_MANEUVER_WHEN_SAFE = "change_maneuver_when_safe"


@dataclass(frozen=True, slots=True)
class ParkingAssessment:
    """经过校验且可序列化为 JSON 的泊车风险评估。"""

    schema_version: str
    risk_level: RiskLevel
    events: tuple[ParkingRiskEvent, ...]
    evidence: tuple[str, ...]
    driver_advice: tuple[DriverAdvice, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ParkingAssessment":
        """解析并校验严格的 JSON 兼容映射。"""
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
        """使用冻结字段名返回 JSON 兼容表示。"""
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
