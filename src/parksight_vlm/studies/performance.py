"""Latency, throughput, memory, power, and temperature aggregation."""

from __future__ import annotations

import math
from collections.abc import Sequence

from parksight_vlm.inference import InferenceRecord

from .model import PerformanceMetrics, PercentileSummary


def compute_performance_metrics(records: Sequence[InferenceRecord]) -> PerformanceMetrics:
    """Aggregate only successful executions; failures remain in failure_summary."""
    successful_records = [record for record in records if record.succeeded]
    stage_values: dict[str, list[float]] = {}
    for record in successful_records:
        for stage_name, value in record.stage_timings.to_mapping().items():
            if value is not None:
                stage_values.setdefault(stage_name, []).append(value)

    stage_latency_ms = {
        stage_name: _summarize(values) for stage_name, values in stage_values.items()
    }
    cold_start_ms = None
    if successful_records:
        cold_start_ms = successful_records[0].stage_timings.end_to_end_ms

    total_tokens = 0
    total_decode_ms = 0.0
    for record in successful_records:
        if record.output_tokens is None or record.stage_timings.decode_ms is None:
            continue
        total_tokens += record.output_tokens
        total_decode_ms += record.stage_timings.decode_ms
    tokens_per_second = None
    if total_decode_ms > 0:
        tokens_per_second = total_tokens / (total_decode_ms / 1000.0)

    memory_values = _resource_values(successful_records, "peak_memory_mb")
    power_values = _resource_values(successful_records, "average_power_w")
    temperature_values = _resource_values(successful_records, "peak_temperature_c")
    return PerformanceMetrics(
        successful_sample_count=len(successful_records),
        cold_start_ms=cold_start_ms,
        stage_latency_ms=stage_latency_ms,
        tokens_per_second=tokens_per_second,
        peak_memory_mb=max(memory_values) if memory_values else None,
        average_power_w=sum(power_values) / len(power_values) if power_values else None,
        peak_temperature_c=max(temperature_values) if temperature_values else None,
    )


def _summarize(values: list[float]) -> PercentileSummary:
    ordered_values = sorted(values)
    return PercentileSummary(
        count=len(ordered_values),
        p50=_percentile(ordered_values, 0.50),
        p90=_percentile(ordered_values, 0.90),
        p99=_percentile(ordered_values, 0.99),
    )


def _percentile(values: list[float], quantile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return values[lower_index]
    weight = position - lower_index
    return values[lower_index] * (1.0 - weight) + values[upper_index] * weight


def _resource_values(records: list[InferenceRecord], field_name: str) -> list[float]:
    values = []
    for record in records:
        value = getattr(record.resource_snapshot, field_name)
        if value is not None:
            values.append(value)
    return values
