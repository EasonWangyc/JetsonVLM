"""TensorRT Edge-LLM Runtime Adapter。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol
from urllib.request import Request, urlopen

from parksight_vlm.workload import FrozenWorkload

from .runtime import RiskRuntime, RuntimeGeneration, RuntimeIdentity


class EdgeLlmBackend(Protocol):
    """面向已安装 Jetson Runtime 实现的可执行 Edge-LLM 接口。"""

    def generate(self, *, image_path: Path, workload: FrozenWorkload) -> RuntimeGeneration:
        """调用 engine 并返回原始输出和实测事实。"""


class EdgeLlmHttpBackend:
    """实验性 Edge-LLM OpenAI-compatible server 客户端。"""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000",
        model_name: str = "local",
        timeout_seconds: float = 120.0,
    ) -> None:
        self._endpoint = base_url.rstrip("/") + "/v1/chat/completions"
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds

    def generate(self, *, image_path: Path, workload: FrozenWorkload) -> RuntimeGeneration:
        payload: dict[str, object] = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": workload.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": workload.render_user_prompt()},
                    ],
                },
            ],
            "max_tokens": workload.generation.max_new_tokens,
        }
        if not workload.generation.do_sample:
            payload["temperature"] = 0.0
        request = Request(
            self._endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self._timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        try:
            raw_output = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("Edge-LLM response has no assistant content") from error
        if not isinstance(raw_output, str):
            raise RuntimeError("Edge-LLM assistant content must be a string")
        usage = response_payload.get("usage", {})
        output_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if not isinstance(output_tokens, int):
            output_tokens = None
        return RuntimeGeneration(raw_output=raw_output, output_tokens=output_tokens)


class EdgeLlmRuntime(RiskRuntime):
    """记录 TensorRT Edge-LLM 后端实际执行事实的 Adapter。"""

    def __init__(
        self,
        *,
        data_root: Path,
        backend: EdgeLlmBackend,
        backend_revision: str,
        model_id: str,
        model_revision: str,
        adapter_revision: str,
        precision: str,
    ) -> None:
        super().__init__(
            data_root=data_root,
            identity=RuntimeIdentity(
                backend="tensorrt_edge_llm",
                backend_revision=backend_revision,
                model_id=model_id,
                model_revision=model_revision,
                adapter_revision=adapter_revision,
                precision=precision,
            ),
        )
        self._backend = backend

    def _generate(self, *, image_path: Path, workload: FrozenWorkload) -> RuntimeGeneration:
        return self._backend.generate(image_path=image_path, workload=workload)
