"""泊车风险评估的严格 JSON 契约。"""

from .model import (
    ASSESSMENT_SCHEMA_VERSION,
    AssessmentValidationError,
    DriverAdvice,
    ParkingAssessment,
    ParkingRiskEvent,
    RiskLevel,
)

__all__ = [
    "ASSESSMENT_SCHEMA_VERSION",
    "AssessmentValidationError",
    "DriverAdvice",
    "ParkingAssessment",
    "ParkingRiskEvent",
    "RiskLevel",
]
