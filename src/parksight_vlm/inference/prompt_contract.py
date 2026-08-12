"""比较 Transformers 与 Edge-LLM 的可审计 Prompt 契约。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from parksight_vlm.workload import FrozenWorkload

from .edge_llm import EdgeLlmHttpBackend
from .transformers import build_qwen3_vl_chat_messages


PROMPT_CONTRACT_SCHEMA_VERSION = "parksight_prompt_contract_report_v1"


class ChatTemplateProcessor(Protocol):
    """诊断所需的最小 Transformers Processor 接口。"""

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        """将消息渲染成模型真正接收的文本 Prompt。"""


def build_prompt_contract_report(
    *,
    workload: FrozenWorkload,
    image_path: Path,
    model_source: Path,
    processed_chat_template_path: Path,
    processor: ChatTemplateProcessor,
    model_name: str = "local",
) -> dict[str, Any]:
    """生成不执行模型推理的 Prompt/Template 对齐报告。"""
    image_path = image_path.resolve()
    model_source = model_source.resolve()
    processed_chat_template_path = processed_chat_template_path.resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"找不到诊断图片：{image_path}")
    if not model_source.exists():
        raise FileNotFoundError(f"找不到模型来源：{model_source}")

    template_bytes = processed_chat_template_path.read_bytes()
    try:
        template_payload = json.loads(template_bytes.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError("processed chat template 不是 UTF-8") from error
    except json.JSONDecodeError as error:
        raise ValueError("processed chat template 不是合法 JSON") from error

    transformers_messages = build_qwen3_vl_chat_messages(
        image=str(image_path),
        workload=workload,
    )
    transformers_rendered_prompt = processor.apply_chat_template(
        transformers_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(transformers_rendered_prompt, str):
        raise TypeError("Transformers processor 必须返回文本 Prompt")

    edge_backend = EdgeLlmHttpBackend(model_name=model_name)
    edge_http_request = edge_backend.build_request_payload(
        image_path=image_path,
        workload=workload,
    )
    edge_messages = edge_http_request["messages"]
    if not isinstance(edge_messages, list):
        raise TypeError("Edge-LLM messages 必须是数组")

    system_content = edge_messages[0]["content"]
    user_content = edge_messages[1]["content"]
    if not isinstance(user_content, list):
        raise TypeError("Edge-LLM user content 必须是数组")

    return {
        "schema_version": PROMPT_CONTRACT_SCHEMA_VERSION,
        "workload_identity": workload.identity,
        "image_path": str(image_path),
        "model_source": str(model_source),
        "rendered_user_prompt": workload.render_user_prompt(),
        "transformers_messages": transformers_messages,
        "transformers_rendered_prompt": transformers_rendered_prompt,
        "edge_http_request": edge_http_request,
        "processed_chat_template": {
            "path": str(processed_chat_template_path),
            "bytes": len(template_bytes),
            "sha256": hashlib.sha256(template_bytes).hexdigest(),
            "payload": template_payload,
        },
        "message_contract": {
            "messages_equal": edge_messages == transformers_messages,
            "system_content_type": (
                "array" if isinstance(system_content, list) else type(system_content).__name__
            ),
            "user_content_types": [
                item.get("type") if isinstance(item, dict) else None
                for item in user_content
            ],
        },
    }
