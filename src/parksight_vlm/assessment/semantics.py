"""泊车评估字段之间的可解释语义一致性审计。

该模块不改变 ``ParkingAssessment`` 的 JSON schema，也不把告警转换为模型输出。
它只检查结构化字段之间是否存在明显冲突，供 StudyReport 和人工复盘使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .model import DriverAdvice, ParkingAssessment, ParkingRiskEvent, RiskLevel


class SemanticIssue(str, Enum):
    """可稳定统计的跨字段语义告警。"""

    LOW_RISK_WITH_PREPARE_TO_STOP = "low_risk_with_prepare_to_stop"
    HIGH_RISK_WITHOUT_IMMEDIATE_RESPONSE = "high_risk_without_immediate_response"
    PATH_CONFLICT_WITHOUT_IMMEDIATE_RESPONSE = "path_conflict_without_immediate_response"
    NON_LOW_RISK_WITHOUT_EVENT = "non_low_risk_without_event"


@dataclass(frozen=True, slots=True)
class SemanticAudit:
    """一次已通过 JSON 校验的 assessment 的语义审计结果。"""

    issues: tuple[SemanticIssue, ...]

    @property
    def is_consistent(self) -> bool:
        """是否未发现当前规则覆盖的字段冲突。"""
        return not self.issues

    def to_mapping(self) -> dict[str, object]:
        """返回 JSON 兼容的审计结果。"""
        return {
            "is_consistent": self.is_consistent,
            "issues": [issue.value for issue in self.issues],
        }


_IMMEDIATE_RESPONSE = frozenset(
    {DriverAdvice.YIELD, DriverAdvice.PREPARE_TO_STOP}
)
_PATH_CONFLICT_EVENTS = frozenset(
    {
        ParkingRiskEvent.VRU_NEAR_MANEUVER_PATH,
        ParkingRiskEvent.VEHICLE_NEAR_MANEUVER_PATH,
    }
)


def audit_assessment_semantics(assessment: ParkingAssessment) -> SemanticAudit:
    """检查风险等级、风险事件与驾驶建议之间的明显语义冲突。

    规则是诊断规则而非硬拒绝规则：

    * ``low`` 场景不应给出 ``prepare_to_stop``，因为该建议表示随时准备停车，
      不是“准备泊入车位”；
    * ``high`` 场景，或明确存在行人/车辆近机动路径事件时，应至少包含
      ``yield`` 或 ``prepare_to_stop``；
    * 非 ``low`` 场景通常应有至少一个已定义事件作为风险依据。

    中风险场景可以包含 ``prepare_to_stop``，因此本审计不会把它误报为冲突；
    最终标签是否正确仍由人工参考标注和质量指标判断。
    """
    issues: list[SemanticIssue] = []
    advice = set(assessment.driver_advice)
    events = set(assessment.events)

    if (
        assessment.risk_level == RiskLevel.LOW
        and DriverAdvice.PREPARE_TO_STOP in advice
    ):
        issues.append(SemanticIssue.LOW_RISK_WITH_PREPARE_TO_STOP)

    if (
        assessment.risk_level == RiskLevel.HIGH
        and not advice.intersection(_IMMEDIATE_RESPONSE)
    ):
        issues.append(SemanticIssue.HIGH_RISK_WITHOUT_IMMEDIATE_RESPONSE)

    if (
        events.intersection(_PATH_CONFLICT_EVENTS)
        and not advice.intersection(_IMMEDIATE_RESPONSE)
    ):
        issues.append(SemanticIssue.PATH_CONFLICT_WITHOUT_IMMEDIATE_RESPONSE)

    if assessment.risk_level != RiskLevel.LOW and not events:
        issues.append(SemanticIssue.NON_LOW_RISK_WITHOUT_EVENT)

    return SemanticAudit(issues=tuple(issues))
