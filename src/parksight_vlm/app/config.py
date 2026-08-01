"""用于组合领域模块的严格应用配置。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from parksight_vlm.studies import StudyDefinition
from parksight_vlm.workload import FrozenWorkload


class AppConfigError(ValueError):
    """当 Runtime 或研究应用配置无效时抛出。"""


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    backend: str
    backend_revision: str
    model_id: str
    model_revision: str
    adapter_revision: str
    precision: str
    options: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RuntimeConfig":
        if not isinstance(payload, Mapping):
            raise AppConfigError("runtime must be a mapping")
        expected_fields = {
            "backend",
            "backend_revision",
            "model_id",
            "model_revision",
            "adapter_revision",
            "precision",
            "options",
        }
        _require_exact_fields(payload, expected_fields, "runtime")
        options = payload["options"]
        if not isinstance(options, Mapping):
            raise AppConfigError("runtime.options must be a mapping")
        return cls(
            backend=_parse_text(payload["backend"], "runtime.backend"),
            backend_revision=_parse_text(
                payload["backend_revision"], "runtime.backend_revision"
            ),
            model_id=_parse_text(payload["model_id"], "runtime.model_id"),
            model_revision=_parse_text(
                payload["model_revision"], "runtime.model_revision"
            ),
            adapter_revision=_parse_text(
                payload["adapter_revision"], "runtime.adapter_revision"
            ),
            precision=_parse_text(payload["precision"], "runtime.precision"),
            options=dict(options),
        )


@dataclass(frozen=True, slots=True)
class AppStudyConfig:
    """执行一个 StudyDefinition 所需的路径和 Runtime 设置。"""

    study: StudyDefinition
    runtime: RuntimeConfig
    manifest_path: Path
    annotations_path: Path
    data_root: Path
    output_path: Path

    @classmethod
    def load(cls, path: Path | str) -> "AppStudyConfig":
        config_path = Path(path).resolve()
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise AppConfigError(f"cannot read app study config: {config_path}") from error
        except json.JSONDecodeError as error:
            raise AppConfigError(f"invalid app study JSON: {config_path}") from error
        if not isinstance(payload, Mapping):
            raise AppConfigError("app study config must be a mapping")
        expected_fields = {
            "study_id",
            "workload_path",
            "manifest_path",
            "annotations_path",
            "data_root",
            "output_path",
            "split",
            "repetitions",
            "power_mode",
            "runtime",
        }
        _require_exact_fields(payload, expected_fields, "app study")
        config_root = config_path.parent
        workload = FrozenWorkload.load(
            _resolve_config_path(config_root, payload["workload_path"], "workload_path")
        )
        study = StudyDefinition.from_mapping(
            {
                "study_id": payload["study_id"],
                "split": payload["split"],
                "repetitions": payload["repetitions"],
                "power_mode": payload["power_mode"],
            },
            workload=workload,
        )
        return cls(
            study=study,
            runtime=RuntimeConfig.from_mapping(payload["runtime"]),
            manifest_path=_resolve_config_path(
                config_root, payload["manifest_path"], "manifest_path"
            ),
            annotations_path=_resolve_config_path(
                config_root, payload["annotations_path"], "annotations_path"
            ),
            data_root=_resolve_config_path(
                config_root, payload["data_root"], "data_root"
            ),
            output_path=_resolve_config_path(
                config_root, payload["output_path"], "output_path"
            ),
        )


def _resolve_config_path(root: Path, value: Any, field_name: str) -> Path:
    path_text = _parse_text(value, field_name)
    candidate = Path(path_text)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _require_exact_fields(
    payload: Mapping[str, Any], expected_fields: set[str], context: str
) -> None:
    actual_fields = set(payload)
    if actual_fields != expected_fields:
        raise AppConfigError(
            f"invalid {context} fields; "
            f"missing={sorted(expected_fields - actual_fields)}, "
            f"unexpected={sorted(actual_fields - expected_fields)}"
        )


def _parse_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AppConfigError(f"{field_name} must be a non-blank string")
    return value.strip()
