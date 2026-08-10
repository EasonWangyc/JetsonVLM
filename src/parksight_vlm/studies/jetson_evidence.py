"""从 StudyReport 与 tegrastats 原始日志生成 Jetson 运行证据摘要。"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


_RAM_PATTERN = re.compile(r"\bRAM (\d+)/(\d+)MB")
_SWAP_PATTERN = re.compile(r"\bSWAP (\d+)/(\d+)MB")
_GPU_UTIL_PATTERN = re.compile(r"\bGR3D_FREQ (\d+)%")
_GPU_TEMP_PATTERN = re.compile(r"\bgpu@([0-9.]+)C")
_VDD_IN_PATTERN = re.compile(r"\bVDD_IN (\d+)mW/(\d+)mW")


class JetsonEvidenceError(ValueError):
    """输入证据缺失或格式不满足汇总要求。"""


def summarize_jetson_study(
    *, study_report: Mapping[str, Any], tegrastats_lines: Iterable[str]
) -> dict[str, Any]:
    """分离后端执行事实、严格 JSON 质量和 Jetson 系统遥测。"""
    records = study_report.get("records")
    if not isinstance(records, list):
        raise JetsonEvidenceError("study report records must be an array")

    backend_completed_records = [
        record
        for record in records
        if isinstance(record, Mapping) and isinstance(record.get("raw_output"), str)
    ]
    schema_valid_records = [
        record
        for record in records
        if isinstance(record, Mapping) and isinstance(record.get("assessment"), Mapping)
    ]
    end_to_end_values = _record_numbers(
        backend_completed_records, "stage_timings", "end_to_end_ms"
    )
    output_tokens = _record_numbers(backend_completed_records, "output_tokens")
    total_end_to_end_ms = sum(end_to_end_values)
    aggregate_output_tokens_per_end_to_end_second = None
    if total_end_to_end_ms > 0 and output_tokens:
        aggregate_output_tokens_per_end_to_end_second = (
            sum(output_tokens) / (total_end_to_end_ms / 1000.0)
        )

    telemetry = parse_tegrastats(tegrastats_lines)
    failure_summary = study_report.get("failure_summary", {})
    if not isinstance(failure_summary, Mapping):
        raise JetsonEvidenceError("study report failure_summary must be an object")

    return {
        "schema_version": "parksight_jetson_runtime_summary_v1",
        "study_identity": study_report.get("study_identity"),
        "runtime_execution": {
            "record_count": len(records),
            "backend_completed_record_count": len(backend_completed_records),
            "schema_valid_record_count": len(schema_valid_records),
            "failure_summary": dict(failure_summary),
            "end_to_end_ms": _numeric_summary(end_to_end_values),
            "output_tokens": {
                **_numeric_summary(output_tokens),
                "total": sum(output_tokens),
                "aggregate_output_tokens_per_end_to_end_second": (
                    aggregate_output_tokens_per_end_to_end_second
                ),
            },
        },
        "jetson_telemetry": telemetry,
    }


def parse_tegrastats(lines: Iterable[str]) -> dict[str, Any]:
    """解析 JetPack 6 tegrastats 的关键资源字段。"""
    ram_used_mb: list[float] = []
    ram_total_mb: list[float] = []
    swap_used_mb: list[float] = []
    swap_total_mb: list[float] = []
    gpu_util_percent: list[float] = []
    gpu_temperature_c: list[float] = []
    vdd_in_w: list[float] = []
    parsed_line_count = 0

    for line in lines:
        ram_match = _RAM_PATTERN.search(line)
        swap_match = _SWAP_PATTERN.search(line)
        gpu_util_match = _GPU_UTIL_PATTERN.search(line)
        gpu_temp_match = _GPU_TEMP_PATTERN.search(line)
        vdd_in_match = _VDD_IN_PATTERN.search(line)
        if not all(
            (ram_match, swap_match, gpu_util_match, gpu_temp_match, vdd_in_match)
        ):
            continue
        parsed_line_count += 1
        ram_used_mb.append(float(ram_match.group(1)))
        ram_total_mb.append(float(ram_match.group(2)))
        swap_used_mb.append(float(swap_match.group(1)))
        swap_total_mb.append(float(swap_match.group(2)))
        gpu_util_percent.append(float(gpu_util_match.group(1)))
        gpu_temperature_c.append(float(gpu_temp_match.group(1)))
        vdd_in_w.append(float(vdd_in_match.group(1)) / 1000.0)

    if parsed_line_count == 0:
        raise JetsonEvidenceError("tegrastats log contains no complete telemetry rows")
    return {
        "sample_count": parsed_line_count,
        "ram_total_mb": max(ram_total_mb),
        "ram_used_mb": _numeric_summary(ram_used_mb),
        "swap_total_mb": max(swap_total_mb),
        "swap_used_mb": _numeric_summary(swap_used_mb),
        "gpu_utilization_percent": _numeric_summary(gpu_util_percent),
        "gpu_temperature_c": _numeric_summary(gpu_temperature_c),
        "vdd_in_w": _numeric_summary(vdd_in_w),
    }


def write_jetson_study_summary(
    *, study_report_path: Path, tegrastats_path: Path, output_path: Path
) -> dict[str, Any]:
    """读取原始证据、写入派生摘要，并返回摘要内容。"""
    try:
        study_report = json.loads(study_report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise JetsonEvidenceError(
            f"cannot read study report: {study_report_path}"
        ) from error
    if not isinstance(study_report, Mapping):
        raise JetsonEvidenceError("study report root must be an object")
    try:
        tegrastats_lines = tegrastats_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise JetsonEvidenceError(
            f"cannot read tegrastats log: {tegrastats_path}"
        ) from error

    summary = summarize_jetson_study(
        study_report=study_report,
        tegrastats_lines=tegrastats_lines,
    )
    summary["evidence_sources"] = {
        "study_report": str(study_report_path),
        "tegrastats_log": str(tegrastats_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _record_numbers(
    records: Sequence[Mapping[str, Any]], *field_path: str
) -> list[float]:
    values: list[float] = []
    for record in records:
        value: Any = record
        for field_name in field_path:
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(field_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        values.append(float(value))
    return values


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "maximum": None,
        }
    ordered_values = sorted(values)
    return {
        "count": len(ordered_values),
        "minimum": ordered_values[0],
        "mean": sum(ordered_values) / len(ordered_values),
        "p50": _percentile(ordered_values, 0.50),
        "p90": _percentile(ordered_values, 0.90),
        "p99": _percentile(ordered_values, 0.99),
        "maximum": ordered_values[-1],
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return values[lower_index]
    weight = position - lower_index
    return values[lower_index] * (1.0 - weight) + values[upper_index] * weight
