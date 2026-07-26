"""Runtime adapters and immutable inference execution records."""

from .edge_llm import EdgeLlmBackend, EdgeLlmHttpBackend, EdgeLlmRuntime
from .runtime import (
    InferenceRecord,
    ResourceSnapshot,
    RiskRuntime,
    RuntimeDependencyError,
    RuntimeFailure,
    RuntimeFailureCategory,
    RuntimeGeneration,
    RuntimeIdentity,
    RuntimeRefusalError,
    RuntimeUnsupportedError,
    StageTimings,
)
from .transformers import (
    HuggingFaceQwen3VlBackend,
    TransformersBackend,
    TransformersRuntime,
)

__all__ = [
    "EdgeLlmBackend",
    "EdgeLlmHttpBackend",
    "EdgeLlmRuntime",
    "InferenceRecord",
    "ResourceSnapshot",
    "RiskRuntime",
    "RuntimeDependencyError",
    "RuntimeFailure",
    "RuntimeFailureCategory",
    "RuntimeGeneration",
    "RuntimeIdentity",
    "RuntimeRefusalError",
    "RuntimeUnsupportedError",
    "StageTimings",
    "HuggingFaceQwen3VlBackend",
    "TransformersBackend",
    "TransformersRuntime",
]
