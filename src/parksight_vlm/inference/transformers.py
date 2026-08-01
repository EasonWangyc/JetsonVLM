"""Transformers Runtime Adapter 与 Qwen3-VL 参考后端。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol

from parksight_vlm.workload import FrozenWorkload

from .runtime import (
    ResourceSnapshot,
    RiskRuntime,
    RuntimeDependencyError,
    RuntimeGeneration,
    RuntimeIdentity,
    StageTimings,
)


class TransformersBackend(Protocol): # 类似cpp中类中的虚函数
    """由模型集成层实现的可执行 Transformers 接口。"""

    def generate(self, *, image_path: Path, workload: FrozenWorkload) -> RuntimeGeneration:
        """运行模型并返回原始输出和实测事实。"""


class HuggingFaceQwen3VlBackend:
    """遵循上游 Transformers chat API 的 Qwen3-VL 延迟加载后端。"""

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
        self._model = None # 构造时不立即加载模型
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
                (workload.input_size.width, workload.input_size.height) # 预处理图片，转换成RGB并缩放至指定长×宽
            )
        messages = [ # 遵循 Chat API 组织多模态消息（包含 system 角色、图片和文本 user 提示词）
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
            return_tensors="pt", # 张量字典
        )
        inputs.pop("token_type_ids", None)
        if hasattr(inputs, "to") and hasattr(self._model, "device"):
            inputs = inputs.to(self._model.device)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        if self._torch.cuda.is_available():
            self._torch.cuda.reset_peak_memory_stats()
        generate_start = time.perf_counter()
        with self._torch.inference_mode(): # 纯推理，不计算梯度也不保存反向传播状态
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=workload.generation.max_new_tokens,
                do_sample=workload.generation.do_sample,
            )
        model_generate_ms = (time.perf_counter() - generate_start) * 1000.0
        generated_ids_trimmed = [
            output_ids[len(input_ids) :] # 清除输入prompt对应token
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode( # 由token得到文字
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
        _require_cuda_architecture(torch)
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


def _require_cuda_architecture(torch_module: Any) -> None:
    """拒绝无法在已检测 GPU 上执行 kernel 的 CUDA wheel。"""
    if not torch_module.cuda.is_available():
        return
    major, minor = torch_module.cuda.get_device_capability()
    required_architecture = f"sm_{major}{minor}"
    supported_architectures = tuple(torch_module.cuda.get_arch_list())
    if supported_architectures and required_architecture not in supported_architectures:
        supported_text = ", ".join(supported_architectures)
        raise RuntimeDependencyError(
            f"installed torch {torch_module.__version__} does not include CUDA "
            f"kernels for {required_architecture}; supported architectures: "
            f"{supported_text}"
        )


class TransformersRuntime(RiskRuntime):
    """记录 Transformers 后端执行事实的 Adapter。"""

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
