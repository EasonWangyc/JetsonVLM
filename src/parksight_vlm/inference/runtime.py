"""Backend-independent inference execution and failure recording."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from parksight_vlm.assessment import AssessmentValidationError, ParkingAssessment
from parksight_vlm.casebook import CasebookValidationError, ParkingCase
from parksight_vlm.workload import FrozenWorkload


class RuntimeFailureCategory(str, Enum):
    """Stable failure categories used by studies and reports."""

    INPUT_ERROR = "input_error"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    JSON_PARSE_ERROR = "json_parse_error"
    MODEL_REFUSAL = "model_refusal"
    TIMEOUT = "timeout"
    OUT_OF_MEMORY = "out_of_memory"
    UNSUPPORTED_OPERATOR = "unsupported_operator"
    RUNTIME_ERROR = "runtime_error"


class RuntimeDependencyError(RuntimeError):
    """Raised by an adapter when a required runtime dependency is absent."""


class RuntimeUnsupportedError(RuntimeError):
    """Raised when the runtime cannot execute a model operator or configuration."""


class RuntimeRefusalError(RuntimeError):
    """Raised when a backend explicitly identifies a model refusal."""


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    """Configuration that identifies one executable model runtime."""

    backend: str
    backend_revision: str
    model_id: str
    model_revision: str
    adapter_revision: str
    precision: str

    def __post_init__(self) -> None:
        for field_name in (
            "backend",
            "backend_revision",
            "model_id",
            "model_revision",
            "adapter_revision",
            "precision",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"runtime identity {field_name} must not be blank")

    def to_mapping(self) -> dict[str, str]:
        return {
            "backend": self.backend,
            "backend_revision": self.backend_revision,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "adapter_revision": self.adapter_revision,
            "precision": self.precision,
        }


@dataclass(frozen=True, slots=True)
class StageTimings:
    """Optional stage-level latency facts measured in milliseconds."""

    preprocess_ms: float | None = None
    vision_encode_ms: float | None = None
    model_generate_ms: float | None = None
    prefill_ms: float | None = None
    decode_ms: float | None = None
    end_to_end_ms: float | None = None
    time_to_first_token_ms: float | None = None

    def __post_init__(self) -> None:
        for field_name, value in self.to_mapping().items():
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must not be negative")

    def to_mapping(self) -> dict[str, float | None]:
        return {
            "preprocess_ms": self.preprocess_ms,
            "vision_encode_ms": self.vision_encode_ms,
            "model_generate_ms": self.model_generate_ms,
            "prefill_ms": self.prefill_ms,
            "decode_ms": self.decode_ms,
            "end_to_end_ms": self.end_to_end_ms,
            "time_to_first_token_ms": self.time_to_first_token_ms,
        }


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """Optional runtime resource facts captured by a backend or board probe."""

    peak_memory_mb: float | None = None
    average_power_w: float | None = None
    peak_temperature_c: float | None = None

    def __post_init__(self) -> None:
        for field_name, value in self.to_mapping().items():
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must not be negative")

    def to_mapping(self) -> dict[str, float | None]:
        return {
            "peak_memory_mb": self.peak_memory_mb,
            "average_power_w": self.average_power_w,
            "peak_temperature_c": self.peak_temperature_c,
        }


@dataclass(frozen=True, slots=True)
class RuntimeFailure:
    """A categorized runtime failure preserved instead of forged output."""

    category: RuntimeFailureCategory
    message: str
    exception_type: str

    def to_mapping(self) -> dict[str, str]:
        return {
            "category": self.category.value,
            "message": self.message,
            "exception_type": self.exception_type,
        }


@dataclass(frozen=True, slots=True)
class RuntimeGeneration:
    """Raw backend generation plus facts measured during execution."""

    raw_output: str
    stage_timings: StageTimings = field(default_factory=StageTimings)
    resource_snapshot: ResourceSnapshot = field(default_factory=ResourceSnapshot)
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.raw_output, str):
            raise TypeError("raw_output must be a string")
        if self.output_tokens is not None and self.output_tokens < 0:
            raise ValueError("output_tokens must not be negative")


@dataclass(frozen=True, slots=True)
class InferenceRecord:
    """All available facts from one runtime execution."""

    case_id: str
    runtime_identity: RuntimeIdentity
    workload_identity: str
    started_at_utc: str
    assessment: ParkingAssessment | None
    failure: RuntimeFailure | None
    raw_output: str | None
    stage_timings: StageTimings
    resource_snapshot: ResourceSnapshot
    output_tokens: int | None

    def __post_init__(self) -> None:
        if (self.assessment is None) == (self.failure is None):
            raise ValueError("record must contain exactly one of assessment or failure")

    @property
    def succeeded(self) -> bool:
        return self.assessment is not None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "runtime_identity": self.runtime_identity.to_mapping(),
            "workload_identity": self.workload_identity,
            "started_at_utc": self.started_at_utc,
            "assessment": (
                self.assessment.to_mapping() if self.assessment is not None else None
            ),
            "failure": self.failure.to_mapping() if self.failure is not None else None,
            "raw_output": self.raw_output,
            "stage_timings": self.stage_timings.to_mapping(),
            "resource_snapshot": self.resource_snapshot.to_mapping(),
            "output_tokens": self.output_tokens,
        }


class RiskRuntime(ABC):
    """Template interface that turns backend execution into an inference record."""

    def __init__(self, *, data_root: Path, identity: RuntimeIdentity) -> None:
        self._data_root = data_root
        self._identity = identity

    @property
    def identity(self) -> RuntimeIdentity:
        return self._identity

    def analyze(self, case: ParkingCase, workload: FrozenWorkload) -> InferenceRecord:
        """Execute one case and preserve either validated output or failure facts."""
        started_at_utc = datetime.now(timezone.utc).isoformat()
        start_time = time.perf_counter()
        generation: RuntimeGeneration | None = None
        raw_output: str | None = None
        assessment: ParkingAssessment | None = None
        failure: RuntimeFailure | None = None
        try:
            image_path = case.resolve_image(self._data_root, require_exists=True)
            generation = self._generate(image_path=image_path, workload=workload)
            raw_output = generation.raw_output
            payload = json.loads(raw_output)
            assessment = ParkingAssessment.from_mapping(payload)
        except Exception as error:  # The record is the explicit runtime failure boundary.
            failure = _classify_failure(error)

        end_to_end_ms = (time.perf_counter() - start_time) * 1000.0
        if generation is None:
            timings = StageTimings(end_to_end_ms=end_to_end_ms)
            resources = ResourceSnapshot()
            output_tokens = None
        else:
            timings = replace(generation.stage_timings, end_to_end_ms=end_to_end_ms)
            resources = generation.resource_snapshot
            output_tokens = generation.output_tokens

        return InferenceRecord(
            case_id=case.case_id,
            runtime_identity=self.identity,
            workload_identity=workload.identity,
            started_at_utc=started_at_utc,
            assessment=assessment,
            failure=failure,
            raw_output=raw_output,
            stage_timings=timings,
            resource_snapshot=resources,
            output_tokens=output_tokens,
        )

    @abstractmethod
    def _generate(self, *, image_path: Path, workload: FrozenWorkload) -> RuntimeGeneration:
        """Invoke the concrete backend and return only actually measured facts."""


def _classify_failure(error: Exception) -> RuntimeFailure:
    message = str(error) or error.__class__.__name__
    lowered_message = message.lower()
    if isinstance(error, CasebookValidationError):
        category = RuntimeFailureCategory.INPUT_ERROR
    elif isinstance(error, RuntimeDependencyError):
        category = RuntimeFailureCategory.DEPENDENCY_UNAVAILABLE
    elif isinstance(error, (json.JSONDecodeError, AssessmentValidationError)):
        category = RuntimeFailureCategory.JSON_PARSE_ERROR
    elif isinstance(error, RuntimeRefusalError):
        category = RuntimeFailureCategory.MODEL_REFUSAL
    elif isinstance(error, TimeoutError):
        category = RuntimeFailureCategory.TIMEOUT
    elif isinstance(error, MemoryError) or "out of memory" in lowered_message:
        category = RuntimeFailureCategory.OUT_OF_MEMORY
    elif isinstance(error, RuntimeUnsupportedError) or "unsupported operator" in lowered_message:
        category = RuntimeFailureCategory.UNSUPPORTED_OPERATOR
    else:
        category = RuntimeFailureCategory.RUNTIME_ERROR
    return RuntimeFailure(
        category=category,
        message=message,
        exception_type=error.__class__.__name__,
    )
