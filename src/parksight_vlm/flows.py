"""训练、合并、量化、导出和 engine 构建命令的可审计计划。"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class FlowValidationError(ValueError):
    """当外部模型流程计划不完整或不一致时抛出。"""


FLOW_STAGES = frozenset(
    {
        "train_lora",
        "merge_lora",
        "quantize_model",
        "export_model",
        "build_engine",
    }
)


@dataclass(frozen=True, slots=True)
class ExternalFlowPlan:
    """经过审核的命令，以及使其可审计的输入和输出。"""

    flow_id: str
    stage: str
    working_directory: Path
    command: tuple[str, ...]
    required_inputs: tuple[Path, ...]
    expected_outputs: tuple[Path, ...]
    record_path: Path
    log_path: Path

    @classmethod
    def load(cls, path: Path | str) -> "ExternalFlowPlan":
        plan_path = Path(path).resolve()
        try:
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise FlowValidationError(f"cannot read flow plan: {plan_path}") from error
        except json.JSONDecodeError as error:
            raise FlowValidationError(f"invalid flow plan JSON: {plan_path}") from error
        return cls.from_mapping(payload, config_root=plan_path.parent)

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, config_root: Path
    ) -> "ExternalFlowPlan":
        if not isinstance(payload, Mapping):
            raise FlowValidationError("flow plan must be a mapping")
        expected_fields = {
            "flow_id",
            "stage",
            "working_directory",
            "command",
            "required_inputs",
            "expected_outputs",
            "record_path",
            "log_path",
        }
        actual_fields = set(payload)
        if actual_fields != expected_fields:
            raise FlowValidationError(
                "invalid flow plan fields; "
                f"missing={sorted(expected_fields - actual_fields)}, "
                f"unexpected={sorted(actual_fields - expected_fields)}"
            )
        stage = _parse_text(payload["stage"], "stage")
        if stage not in FLOW_STAGES:
            raise FlowValidationError(
                f"stage must be one of {sorted(FLOW_STAGES)}"
            )
        command = _parse_text_sequence(payload["command"], "command")
        required_inputs = _parse_path_sequence(
            payload["required_inputs"], config_root, "required_inputs"
        )
        expected_outputs = _parse_path_sequence(
            payload["expected_outputs"], config_root, "expected_outputs"
        )
        if not expected_outputs:
            raise FlowValidationError("expected_outputs must not be empty")
        record_path = _resolve_path(
            config_root, payload["record_path"], "record_path"
        )
        log_path = _resolve_path(config_root, payload["log_path"], "log_path")
        protected_paths = set(required_inputs) | set(expected_outputs)
        if record_path == log_path:
            raise FlowValidationError("record_path and log_path must be different")
        if record_path in protected_paths or log_path in protected_paths:
            raise FlowValidationError(
                "record_path and log_path must not overlap inputs or outputs"
            )
        return cls(
            flow_id=_parse_text(payload["flow_id"], "flow_id"),
            stage=stage,
            working_directory=_resolve_path(
                config_root, payload["working_directory"], "working_directory"
            ),
            command=command,
            required_inputs=required_inputs,
            expected_outputs=expected_outputs,
            record_path=record_path,
            log_path=log_path,
        )

    @property
    def identity(self) -> str:
        normalized = json.dumps(
            self.plan_mapping(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(normalized).hexdigest()

    def plan_mapping(self) -> dict[str, Any]:
        return {
            "flow_id": self.flow_id,
            "stage": self.stage,
            "working_directory": str(self.working_directory),
            "command": list(self.command),
            "required_inputs": [str(path) for path in self.required_inputs],
            "expected_outputs": [str(path) for path in self.expected_outputs],
            "record_path": str(self.record_path),
            "log_path": str(self.log_path),
        }

    def readiness_mapping(self) -> dict[str, Any]:
        executable = self.command[0]
        executable_found = (
            Path(executable).is_file()
            if Path(executable).is_absolute()
            else shutil.which(executable) is not None
        )
        missing_inputs = [
            str(path) for path in self.required_inputs if not path.exists()
        ]
        preexisting_outputs = [
            str(path) for path in self.expected_outputs if path.exists()
        ]
        record_path_available = not self.record_path.exists()
        log_path_available = not self.log_path.exists()
        return {
            "plan_identity": self.identity,
            "plan": self.plan_mapping(),
            "ready": (
                self.working_directory.is_dir()
                and executable_found
                and not missing_inputs
                and not preexisting_outputs
                and record_path_available
                and log_path_available
            ),
            "working_directory_exists": self.working_directory.is_dir(),
            "executable_found": executable_found,
            "missing_inputs": missing_inputs,
            "preexisting_outputs": preexisting_outputs,
            "record_path_available": record_path_available,
            "log_path_available": log_path_available,
        }

    def execute(self) -> "ExternalFlowResult":
        """仅在调用方明确选择执行路径后运行。"""
        readiness = self.readiness_mapping()
        if not readiness["ready"]:
            raise FlowValidationError(
                "flow is not ready: "
                f"working_directory_exists={readiness['working_directory_exists']}, "
                f"executable_found={readiness['executable_found']}, "
                f"missing_inputs={readiness['missing_inputs']}, "
                f"preexisting_outputs={readiness['preexisting_outputs']}, "
                f"record_path_available={readiness['record_path_available']}, "
                f"log_path_available={readiness['log_path_available']}"
            )
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        started_at = _utc_now()
        launch_error: str | None = None
        exit_code: int | None = None
        with self.log_path.open("w", encoding="utf-8") as log_file:
            try:
                completed = subprocess.run(
                    self.command,
                    cwd=self.working_directory,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                    text=True,
                )
                exit_code = completed.returncode
            except OSError as error:
                launch_error = f"{type(error).__name__}: {error}"
                log_file.write(launch_error + "\n")
        finished_at = _utc_now()
        output_status = {
            str(path): path.exists() for path in self.expected_outputs
        }
        succeeded = (
            launch_error is None
            and exit_code == 0
            and all(output_status.values())
        )
        result = ExternalFlowResult(
            plan_identity=self.identity,
            flow_id=self.flow_id,
            stage=self.stage,
            status="succeeded" if succeeded else "failed",
            started_at=started_at,
            finished_at=finished_at,
            exit_code=exit_code,
            launch_error=launch_error,
            output_status=output_status,
            log_path=self.log_path,
        )
        self.record_path.write_text(
            json.dumps(result.to_mapping(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return result


@dataclass(frozen=True, slots=True)
class ExternalFlowResult:
    plan_identity: str
    flow_id: str
    stage: str
    status: str
    started_at: str
    finished_at: str
    exit_code: int | None
    launch_error: str | None
    output_status: Mapping[str, bool]
    log_path: Path

    def to_mapping(self) -> dict[str, Any]:
        return {
            "plan_identity": self.plan_identity,
            "flow_id": self.flow_id,
            "stage": self.stage,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "launch_error": self.launch_error,
            "output_status": dict(self.output_status),
            "log_path": str(self.log_path),
        }


def _parse_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FlowValidationError(f"{field_name} must be a non-blank string")
    return value.strip()


def _parse_text_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FlowValidationError(f"{field_name} must be an array of strings")
    parsed = tuple(
        _parse_text(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )
    if not parsed:
        raise FlowValidationError(f"{field_name} must not be empty")
    return parsed


def _parse_path_sequence(
    value: Any, config_root: Path, field_name: str
) -> tuple[Path, ...]:
    paths = tuple(
        _resolve_path(config_root, item, f"{field_name}[{index}]")
        for index, item in enumerate(_parse_sequence(value, field_name))
    )
    if len(set(paths)) != len(paths):
        raise FlowValidationError(f"{field_name} must not contain duplicates")
    return paths


def _parse_sequence(value: Any, field_name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FlowValidationError(f"{field_name} must be an array")
    return value


def _resolve_path(root: Path, value: Any, field_name: str) -> Path:
    candidate = Path(_parse_text(value, field_name))
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
