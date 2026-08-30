"""根据推理记录和参考标注计算任务质量指标。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from parksight_vlm.assessment import (
    DriverAdvice,
    ParkingAssessment,
    ParkingRiskEvent,
    RiskLevel,
    audit_assessment_semantics,
)
from parksight_vlm.inference import InferenceRecord

from .model import QualityMetrics, StudyValidationError


def compute_quality_metrics(
    records: Sequence[InferenceRecord],
    references: Mapping[str, ParkingAssessment],
) -> QualityMetrics:
    """基于全部执行计算严格 JSON 有效率和任务指标。"""
    if not records:
        raise StudyValidationError("quality metrics require at least one inference record")

    valid_count = 0
    risk_level_correct = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    unsafe_count = 0
    semantic_valid_count = 0
    semantic_consistent_count = 0
    semantic_issue_counts: dict[str, int] = {}
    event_errors: dict[str, dict[str, int]] = {
        event.value: {"false_positive": 0, "false_negative": 0}
        for event in ParkingRiskEvent
    }
    event_counts: dict[str, dict[str, int]] = {
        event.value: {"support": 0, "true_positive": 0, "false_positive": 0, "false_negative": 0}
        for event in ParkingRiskEvent
    }

    for record in records:
        try:
            reference = references[record.case_id]
        except KeyError as error:
            raise StudyValidationError(
                f"missing reference assessment for case_id: {record.case_id!r}"
            ) from error
        predicted = record.assessment
        if predicted is not None:
            valid_count += 1
            if predicted.risk_level == reference.risk_level:
                risk_level_correct += 1
            semantic_valid_count += 1
            semantic_audit = audit_assessment_semantics(predicted)
            if semantic_audit.is_consistent:
                semantic_consistent_count += 1
            for issue in semantic_audit.issues:
                semantic_issue_counts[issue.value] = (
                    semantic_issue_counts.get(issue.value, 0) + 1
                )

        reference_events = set(reference.events)
        predicted_events = set(predicted.events) if predicted is not None else set()
        true_positive += len(reference_events & predicted_events)
        false_positive += len(predicted_events - reference_events)
        false_negative += len(reference_events - predicted_events)
        for event in ParkingRiskEvent:
            event_name = event.value
            event_in_reference = event in reference_events
            event_in_prediction = event in predicted_events
            event_counts[event_name]["support"] += int(event_in_reference)
            event_counts[event_name]["true_positive"] += int(
                event_in_reference and event_in_prediction
            )
            event_counts[event_name]["false_positive"] += int(
                not event_in_reference and event_in_prediction
            )
            event_counts[event_name]["false_negative"] += int(
                event_in_reference and not event_in_prediction
            )
        for event in predicted_events - reference_events:
            event_errors[event.value]["false_positive"] += 1
        for event in reference_events - predicted_events:
            event_errors[event.value]["false_negative"] += 1

        if _is_unsafe_advice(reference, predicted):
            unsafe_count += 1

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    event_metrics: dict[str, dict[str, int | float]] = {}
    event_f1_values: list[float] = []
    for event in ParkingRiskEvent:
        counts = event_counts[event.value]
        event_precision = _safe_divide(
            counts["true_positive"],
            counts["true_positive"] + counts["false_positive"],
        )
        event_recall = _safe_divide(
            counts["true_positive"],
            counts["true_positive"] + counts["false_negative"],
        )
        event_f1 = _safe_divide(
            2.0 * event_precision * event_recall,
            event_precision + event_recall,
        )
        event_f1_values.append(event_f1)
        event_metrics[event.value] = {
            **counts,
            "precision": event_precision,
            "recall": event_recall,
            "f1": event_f1,
        }
    sample_count = len(records)
    return QualityMetrics(
        sample_count=sample_count,
        json_validity_rate=valid_count / sample_count,
        risk_level_accuracy=risk_level_correct / sample_count,
        event_micro_precision=precision,
        event_micro_recall=recall,
        event_micro_f1=f1,
        event_macro_f1=sum(event_f1_values) / len(event_f1_values),
        unsafe_advice_rate=unsafe_count / sample_count,
        semantic_consistency_rate=(
            semantic_consistent_count / semantic_valid_count
            if semantic_valid_count
            else 0.0
        ),
        event_errors=event_errors,
        event_metrics=event_metrics,
        semantic_issue_counts=dict(sorted(semantic_issue_counts.items())),
    )


def _is_unsafe_advice(
    reference: ParkingAssessment, predicted: ParkingAssessment | None
) -> bool:
    """应用质量研究所使用的 v1 保守安全策略。"""
    risk_requires_stop = reference.risk_level == RiskLevel.HIGH
    yield_events = {
        ParkingRiskEvent.VRU_NEAR_MANEUVER_PATH,
        ParkingRiskEvent.VEHICLE_NEAR_MANEUVER_PATH,
    }
    event_requires_yield = bool(set(reference.events) & yield_events)
    if not risk_requires_stop and not event_requires_yield:
        return False
    if predicted is None:
        return True
    safe_responses = {DriverAdvice.YIELD, DriverAdvice.PREPARE_TO_STOP}
    return not bool(set(predicted.driver_advice) & safe_responses)


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
