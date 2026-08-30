"""泊车风险评估的严格 JSON 契约。"""

from .model import (
    ASSESSMENT_SCHEMA_VERSION,
    AssessmentValidationError,
    DriverAdvice,
    ParkingAssessment,
    ParkingRiskEvent,
    RiskLevel,
)
from .semantics import SemanticAudit, SemanticIssue, audit_assessment_semantics

__all__ = [
    "ASSESSMENT_SCHEMA_VERSION",
    "AssessmentValidationError",
    "DriverAdvice",
    "ParkingAssessment",
    "ParkingRiskEvent",
    "RiskLevel",
    "SemanticAudit",
    "SemanticIssue",
    "audit_assessment_semantics",
]
