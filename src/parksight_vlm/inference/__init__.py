"""Runtime Adapter 与不可变推理执行记录。"""

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
