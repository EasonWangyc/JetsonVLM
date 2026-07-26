"""Task-quality metrics computed from inference records and reference labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from parksight_vlm.assessment import (
    DriverAdvice,
    ParkingAssessment,
    ParkingRiskEvent,
    RiskLevel,
)
from parksight_vlm.inference import InferenceRecord

from .model import QualityMetrics, StudyValidationError


def compute_quality_metrics(
    records: Sequence[InferenceRecord],
    references: Mapping[str, ParkingAssessment],
) -> QualityMetrics:
    """Compute strict-JSON validity and task metrics over all executions."""
    if not records:
        raise StudyValidationError("quality metrics require at least one inference record")

    valid_count = 0
    risk_level_correct = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    unsafe_count = 0
    event_errors: dict[str, dict[str, int]] = {
        event.value: {"false_positive": 0, "false_negative": 0}
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

        reference_events = set(reference.events)
        predicted_events = set(predicted.events) if predicted is not None else set()
        true_positive += len(reference_events & predicted_events)
        false_positive += len(predicted_events - reference_events)
        false_negative += len(reference_events - predicted_events)
        for event in predicted_events - reference_events:
            event_errors[event.value]["false_positive"] += 1
        for event in reference_events - predicted_events:
            event_errors[event.value]["false_negative"] += 1

        if _is_unsafe_advice(reference, predicted):
            unsafe_count += 1

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    sample_count = len(records)
    return QualityMetrics(
        sample_count=sample_count,
        json_validity_rate=valid_count / sample_count,
        risk_level_accuracy=risk_level_correct / sample_count,
        event_micro_precision=precision,
        event_micro_recall=recall,
        event_micro_f1=f1,
        unsafe_advice_rate=unsafe_count / sample_count,
        event_errors=event_errors,
    )


def _is_unsafe_advice(
    reference: ParkingAssessment, predicted: ParkingAssessment | None
) -> bool:
    """Apply the v1 conservative safety policy used by the quality study."""
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
