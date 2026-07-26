"""Compose runtime adapters from validated application configuration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from parksight_vlm.inference import (
    EdgeLlmHttpBackend,
    EdgeLlmRuntime,
    HuggingFaceQwen3VlBackend,
    RiskRuntime,
    TransformersRuntime,
)

from .config import AppConfigError, RuntimeConfig


def build_runtime(config: RuntimeConfig, *, data_root: Path) -> RiskRuntime:
    """Build a runtime without loading model weights until its first execution."""
    if (
        config.backend_revision == "main"
        or config.backend_revision.startswith("replace-with-")
    ):
        raise AppConfigError(
            "runtime.backend_revision must identify the installed backend version "
            "or immutable commit"
        )
    if (
        config.model_revision == "main"
        or config.model_revision.startswith("replace-with-")
    ):
        raise AppConfigError(
            "runtime.model_revision must be an immutable model commit, not a "
            "branch or template placeholder"
        )
    if config.backend == "transformers":
        _require_allowed_options(
            config.options,
            {"device_map", "dtype", "attn_implementation"},
            "transformers",
        )
        backend = HuggingFaceQwen3VlBackend(
            model_id=config.model_id,
            model_revision=config.model_revision,
            device_map=_option_text(config.options, "device_map", "auto"),
            dtype=_option_text(config.options, "dtype", "auto"),
            attn_implementation=_option_text(
                config.options, "attn_implementation", "sdpa"
            ),
        )
        return TransformersRuntime(
            data_root=data_root,
            backend=backend,
            backend_revision=config.backend_revision,
            model_id=config.model_id,
            model_revision=config.model_revision,
            adapter_revision=config.adapter_revision,
            precision=config.precision,
        )
    if config.backend == "tensorrt_edge_llm_http":
        _require_allowed_options(
            config.options,
            {"base_url", "model_name", "timeout_seconds"},
            "tensorrt_edge_llm_http",
        )
        timeout_seconds = config.options.get("timeout_seconds", 120.0)
        if isinstance(timeout_seconds, bool) or not isinstance(
            timeout_seconds, (int, float)
        ):
            raise AppConfigError("runtime.options.timeout_seconds must be numeric")
        backend = EdgeLlmHttpBackend(
            base_url=_option_text(config.options, "base_url", "http://127.0.0.1:8000"),
            model_name=_option_text(config.options, "model_name", "local"),
            timeout_seconds=float(timeout_seconds),
        )
        return EdgeLlmRuntime(
            data_root=data_root,
            backend=backend,
            backend_revision=config.backend_revision,
            model_id=config.model_id,
            model_revision=config.model_revision,
            adapter_revision=config.adapter_revision,
            precision=config.precision,
        )
    raise AppConfigError(f"unsupported runtime backend: {config.backend!r}")


def _require_allowed_options(
    options: Mapping[str, Any], allowed_options: set[str], backend: str
) -> None:
    unexpected_options = set(options) - allowed_options
    if unexpected_options:
        raise AppConfigError(
            f"unsupported {backend} options: {sorted(unexpected_options)}"
        )
def _option_text(options: Mapping[str, Any], key: str, default: str) -> str:
    value = options.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise AppConfigError(f"runtime.options.{key} must be a non-blank string")
    return value.strip()
