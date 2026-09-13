"""比较两份 TensorRT StudyReport 的阶段性能、资源和质量指标。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_IDENTITY_FIELDS = (
    "workload_identity",
    "split",
    "power_mode",
)
_RUNTIME_FIELDS = (
    "backend",
    "backend_revision",
    "model_id",
    "model_revision",
    "adapter_revision",
    "precision",
)
_STAGES = (
    "vision_encode_ms",
    "model_generate_ms",
    "prefill_ms",
    "time_to_first_token_ms",
    "decode_ms",
    "end_to_end_ms",
)
_QUALITY_FIELDS = (
    "json_validity_rate",
    "risk_level_accuracy",
    "event_micro_f1",
)
_PERFORMANCE_FIELDS = (
    "tokens_per_second",
    "aggregate_output_tokens_per_end_to_end_second",
    "peak_memory_mb",
    "average_power_w",
    "peak_temperature_c",
)


def compare_study_reports(
    baseline_path: Path | str,
    candidate_path: Path | str,
) -> dict[str, Any]:
    """Return an auditable comparison for two StudyReport JSON files."""
    baseline = _load_report(baseline_path)
    candidate = _load_report(candidate_path)
    baseline_identity = _mapping(baseline.get("study_identity"), "baseline.study_identity")
    candidate_identity = _mapping(candidate.get("study_identity"), "candidate.study_identity")
    baseline_performance = _mapping(
        baseline.get("performance_metrics"), "baseline.performance_metrics"
    )
    candidate_performance = _mapping(
        candidate.get("performance_metrics"), "candidate.performance_metrics"
    )
    baseline_quality = _mapping(baseline.get("quality_metrics"), "baseline.quality_metrics")
    candidate_quality = _mapping(candidate.get("quality_metrics"), "candidate.quality_metrics")

    identity = _compare_fields(baseline_identity, candidate_identity, _IDENTITY_FIELDS)
    baseline_runtime = _mapping(
        baseline_identity.get("runtime_identity"),
        "baseline.study_identity.runtime_identity",
    )
    candidate_runtime = _mapping(
        candidate_identity.get("runtime_identity"),
        "candidate.study_identity.runtime_identity",
    )
    runtime = _compare_fields(baseline_runtime, candidate_runtime, _RUNTIME_FIELDS)
    stages = {
        stage: _compare_summary(
            baseline_performance.get("stage_latency_ms", {}).get(stage)
            if isinstance(baseline_performance.get("stage_latency_ms"), Mapping)
            else None,
            candidate_performance.get("stage_latency_ms", {}).get(stage)
            if isinstance(candidate_performance.get("stage_latency_ms"), Mapping)
            else None,
        )
        for stage in _STAGES
    }
    quality = {
        field: _compare_scalar(baseline_quality.get(field), candidate_quality.get(field))
        for field in _QUALITY_FIELDS
    }
    performance = {
        field: _compare_scalar(
            baseline_performance.get(field), candidate_performance.get(field)
        )
        for field in _PERFORMANCE_FIELDS
    }
    baseline_count = baseline_performance.get("backend_completed_sample_count")
    candidate_count = candidate_performance.get("backend_completed_sample_count")
    return {
        "schema_version": "parksight_tensorrt_study_comparison_v1",
        "inputs": {
            "baseline": str(Path(baseline_path)),
            "candidate": str(Path(candidate_path)),
        },
        "match": {
            "identity": identity,
            "runtime": runtime,
            "comparable": identity["comparable"] and runtime["comparable"],
        },
        "execution": {
            "baseline_repetitions": baseline_identity.get("repetitions"),
            "candidate_repetitions": candidate_identity.get("repetitions"),
            "baseline_backend_completed_sample_count": baseline_count,
            "candidate_backend_completed_sample_count": candidate_count,
            "stage_latency_ms": stages,
            "performance": performance,
        },
        "quality": quality,
        "evidence_boundary": (
            "This compares serialized StudyReport metrics only. It does not attribute a gain to "
            "an operator, tactic, kernel, KV-cache layout, or CUDA Graph; use Engine Inspector "
            "and Nsight evidence for attribution. Missing fields remain unverified."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = compare_study_reports(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


def _load_report(path: Path | str) -> Mapping[str, Any]:
    report_path = Path(path)
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid TensorRT StudyReport: {report_path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"TensorRT StudyReport must be an object: {report_path}")
    return payload


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _compare_fields(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    compared = {
        field: {"baseline": baseline.get(field), "candidate": candidate.get(field)}
        for field in fields
    }
    mismatches = {
        field: values
        for field, values in compared.items()
        if values["baseline"] != values["candidate"]
    }
    missing = [
        field
        for field, values in compared.items()
        if values["baseline"] is None or values["candidate"] is None
    ]
    return {
        "status": "mismatch" if mismatches else ("unverified" if missing else "matched"),
        "comparable": not mismatches and not missing,
        "compared_fields": compared,
        "mismatches": mismatches,
        "missing_fields": missing,
    }


def _compare_summary(baseline: Any, candidate: Any) -> dict[str, Any]:
    if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
        return {
            "status": "unverified",
            "p50": _compare_scalar(None, None),
            "p90": _compare_scalar(None, None),
            "p99": _compare_scalar(None, None),
            "baseline_count": baseline.get("count") if isinstance(baseline, Mapping) else None,
            "candidate_count": candidate.get("count") if isinstance(candidate, Mapping) else None,
        }
    result = {
        percentile: _compare_scalar(baseline.get(percentile), candidate.get(percentile))
        for percentile in ("p50", "p90", "p99")
    }
    result["status"] = (
        "matched"
        if all(item["improvement"] is not None for item in result.values())
        else "unverified"
    )
    result["baseline_count"] = baseline.get("count")
    result["candidate_count"] = candidate.get("count")
    return result


def _compare_scalar(baseline: Any, candidate: Any) -> dict[str, Any]:
    if not _is_number(baseline) or not _is_number(candidate):
        return {
            "baseline": baseline,
            "candidate": candidate,
            "improvement": None,
            "speedup": None,
        }
    baseline_float = float(baseline)
    candidate_float = float(candidate)
    return {
        "baseline": baseline_float,
        "candidate": candidate_float,
        "improvement": (
            1.0 - candidate_float / baseline_float if baseline_float > 0 else None
        ),
        "speedup": (
            baseline_float / candidate_float if candidate_float > 0 else None
        ),
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


if __name__ == "__main__":
    raise SystemExit(main())
