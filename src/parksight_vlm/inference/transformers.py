"""Transformers runtime adapter and Qwen3-VL reference backend."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

from parksight_vlm.workload import FrozenWorkload

from .runtime import (
    ResourceSnapshot,
    RiskRuntime,
    RuntimeDependencyError,
    RuntimeGeneration,
    RuntimeIdentity,
    StageTimings,
)


class TransformersBackend(Protocol):
    """Executable Transformers seam implemented by the model integration layer."""

    def generate(self, *, image_path: Path, workload: FrozenWorkload) -> RuntimeGeneration:
        """Run the model and return its raw output and measured facts."""


class HuggingFaceQwen3VlBackend:
    """Lazy Qwen3-VL backend following the upstream Transformers chat API."""

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        device_map: str = "auto",
        dtype: str = "auto",
        attn_implementation: str = "sdpa",
    ) -> None:
        self._model_id = model_id
        self._model_revision = model_revision
        self._device_map = device_map
        self._dtype = dtype
        self._attn_implementation = attn_implementation
        self._processor = None
        self._model = None
        self._torch = None

    def generate(self, *, image_path: Path, workload: FrozenWorkload) -> RuntimeGeneration:
        self._ensure_loaded()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None

        preprocess_start = time.perf_counter()
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeDependencyError(
                "Pillow is required by HuggingFaceQwen3VlBackend"
            ) from error
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB").resize(
                (workload.input_size.width, workload.input_size.height)
            )
        messages = [
            {"role": "system", "content": workload.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": workload.user_prompt},
                ],
            },
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        if hasattr(inputs, "to") and hasattr(self._model, "device"):
            inputs = inputs.to(self._model.device)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        if self._torch.cuda.is_available():
            self._torch.cuda.reset_peak_memory_stats()
        generate_start = time.perf_counter()
        with self._torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=workload.generation.max_new_tokens,
                do_sample=workload.generation.do_sample,
            )
        model_generate_ms = (time.perf_counter() - generate_start) * 1000.0
        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        output_tokens = sum(len(token_ids) for token_ids in generated_ids_trimmed)

        peak_memory_mb = None
        if self._torch.cuda.is_available():
            peak_memory_mb = self._torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        return RuntimeGeneration(
            raw_output=output_text,
            stage_timings=StageTimings(
                preprocess_ms=preprocess_ms,
                model_generate_ms=model_generate_ms,
            ),
            resource_snapshot=ResourceSnapshot(peak_memory_mb=peak_memory_mb),
            output_tokens=output_tokens,
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise RuntimeDependencyError(
                "torch and transformers with Qwen3-VL support are required"
            ) from error
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(
            self._model_id,
            revision=self._model_revision,
        )
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self._model_id,
            revision=self._model_revision,
            device_map=self._device_map,
            dtype=self._dtype,
            attn_implementation=self._attn_implementation,
        )
        self._model.eval()


class TransformersRuntime(RiskRuntime):
    """Adapter that records executions from a Transformers backend."""

    def __init__(
        self,
        *,
        data_root: Path,
        backend: TransformersBackend,
        backend_revision: str,
        model_id: str,
        model_revision: str,
        adapter_revision: str = "none",
        precision: str = "bf16",
    ) -> None:
        super().__init__(
            data_root=data_root,
            identity=RuntimeIdentity(
                backend="transformers",
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
