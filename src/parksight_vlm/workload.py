"""所有推理 Runtime 共用的冻结任务配置。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from parksight_vlm.assessment import (
    ASSESSMENT_SCHEMA_VERSION,
    DriverAdvice,
    ParkingRiskEvent,
)


class WorkloadValidationError(ValueError):
    """当 workload 配置不完整或不一致时抛出。"""


@dataclass(frozen=True, slots=True)
class InputSize:
    """提供给 Runtime Adapter 的冻结图片尺寸。"""

    width: int
    height: int


@dataclass(frozen=True, slots=True)
class GenerationParameters:
    """workload 使用的后端无关文本生成参数。"""

    max_new_tokens: int
    do_sample: bool


@dataclass(frozen=True, slots=True)
class FrozenWorkload:
    """独立于模型 Runtime、带版本的泊车风险任务定义。"""

    workload_id: str
    schema_version: str
    system_prompt: str
    user_prompt: str
    input_size: InputSize
    generation: GenerationParameters
    risk_events: tuple[ParkingRiskEvent, ...]
    driver_advice: tuple[DriverAdvice, ...]

    @classmethod
    def load(cls, path: Path | str) -> "FrozenWorkload":
        """加载 UTF-8 编码的 JSON workload 文件。"""
        workload_path = Path(path)
        try:
            payload = json.loads(workload_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise WorkloadValidationError(
                f"cannot read workload file: {workload_path}"
            ) from error
        except json.JSONDecodeError as error:
            raise WorkloadValidationError(
                f"invalid workload JSON: {workload_path}"
            ) from error
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FrozenWorkload":
        """解析并校验严格的 workload 映射。"""
        if not isinstance(payload, Mapping):
            raise WorkloadValidationError("workload payload must be a mapping")
        _require_exact_fields(
            payload,
            {
                "workload_id",
                "schema_version",
                "system_prompt",
                "user_prompt",
                "input_size",
                "generation",
                "risk_events",
                "driver_advice",
            },
            "workload",
        )

        workload_id = _parse_text(payload["workload_id"], "workload_id")
        schema_version = _parse_text(payload["schema_version"], "schema_version")
        if schema_version != ASSESSMENT_SCHEMA_VERSION:
            raise WorkloadValidationError(
                f"unsupported schema_version: {schema_version!r}"
            )

        input_size = _parse_input_size(payload["input_size"])
        generation = _parse_generation(payload["generation"])
        risk_events = _parse_enum_array(
            payload["risk_events"], ParkingRiskEvent, "risk_events"
        )
        driver_advice = _parse_enum_array(
            payload["driver_advice"], DriverAdvice, "driver_advice"
        )
        _require_complete_enum_set(risk_events, ParkingRiskEvent, "risk_events")
        _require_complete_enum_set(driver_advice, DriverAdvice, "driver_advice")

        return cls(
            workload_id=workload_id,
            schema_version=schema_version,
            system_prompt=_parse_text(payload["system_prompt"], "system_prompt"),
            user_prompt=_parse_text(payload["user_prompt"], "user_prompt"),
            input_size=input_size,
            generation=generation,
            risk_events=risk_events,
            driver_advice=driver_advice,
        )

    def to_mapping(self) -> dict[str, Any]:
        """返回规范的 JSON 兼容 workload 表示。"""
        return {
            "workload_id": self.workload_id,
            "schema_version": self.schema_version,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "input_size": {
                "width": self.input_size.width,
                "height": self.input_size.height,
            },
            "generation": {
                "max_new_tokens": self.generation.max_new_tokens,
                "do_sample": self.generation.do_sample,
            },
            "risk_events": [event.value for event in self.risk_events],
            "driver_advice": [advice.value for advice in self.driver_advice],
        }

    @property
    def fingerprint(self) -> str:
        """返回用于报告溯源的稳定 SHA-256 摘要。"""
        canonical_json = json.dumps(
            self.to_mapping(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    @property
    def identity(self) -> str:
        """返回适合写入推理记录的紧凑 workload 身份。"""
        return f"{self.workload_id}@sha256:{self.fingerprint}"


def _parse_input_size(value: Any) -> InputSize:
    if not isinstance(value, Mapping):
        raise WorkloadValidationError("input_size must be a mapping")
    _require_exact_fields(value, {"width", "height"}, "input_size")
    width = _parse_positive_int(value["width"], "input_size.width")
    height = _parse_positive_int(value["height"], "input_size.height")
    return InputSize(width=width, height=height)


def _parse_generation(value: Any) -> GenerationParameters:
    if not isinstance(value, Mapping):
        raise WorkloadValidationError("generation must be a mapping")
    _require_exact_fields(value, {"max_new_tokens", "do_sample"}, "generation")
    max_new_tokens = _parse_positive_int(
        value["max_new_tokens"], "generation.max_new_tokens"
    )
    do_sample = value["do_sample"]
    if not isinstance(do_sample, bool):
        raise WorkloadValidationError("generation.do_sample must be a boolean")
    return GenerationParameters(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )


EnumType = TypeVar("EnumType", bound=ParkingRiskEvent | DriverAdvice)


def _parse_enum_array(
    value: Any, enum_type: type[EnumType], field_name: str
) -> tuple[EnumType, ...]:
    if not isinstance(value, list):
        raise WorkloadValidationError(f"{field_name} must be an array")
    parsed_values: list[EnumType] = []
    for item in value:
        if not isinstance(item, str):
            raise WorkloadValidationError(f"{field_name} entries must be strings")
        try:
            parsed_values.append(enum_type(item))
        except ValueError as error:
            raise WorkloadValidationError(
                f"unsupported {field_name} entry: {item!r}"
            ) from error
    if len(set(parsed_values)) != len(parsed_values):
        raise WorkloadValidationError(f"{field_name} must not contain duplicates")
    return tuple(parsed_values)


def _require_complete_enum_set(
    values: tuple[EnumType, ...], enum_type: type[EnumType], field_name: str
) -> None:
    actual_values = set(values)
    expected_values = set(enum_type)
    missing_values = expected_values - actual_values
    unexpected_values = actual_values - expected_values
    if missing_values or unexpected_values:
        details = []
        if missing_values:
            details.append(
                "missing values: " + str(sorted(value.value for value in missing_values))
            )
        if unexpected_values:
            details.append(
                "unexpected values: "
                + str(sorted(value.value for value in unexpected_values))
            )
        raise WorkloadValidationError(f"{field_name} is incomplete; " + "; ".join(details))


def _require_exact_fields(
    payload: Mapping[str, Any], expected_fields: set[str], context: str
) -> None:
    actual_fields = set(payload)
    missing_fields = expected_fields - actual_fields
    unexpected_fields = actual_fields - expected_fields
    if missing_fields or unexpected_fields:
        details = []
        if missing_fields:
            details.append(f"missing fields: {sorted(missing_fields)}")
        if unexpected_fields:
            details.append(f"unexpected fields: {sorted(unexpected_fields)}")
        raise WorkloadValidationError(f"invalid {context}; " + "; ".join(details))


def _parse_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkloadValidationError(f"{field_name} must be a non-blank string")
    return value.strip()


def _parse_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WorkloadValidationError(f"{field_name} must be a positive integer")
    return value
