"""TensorRT Edge-LLM 部署入口的无硬件行为测试。"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from scripts.build_edgellm_vlm_engines import build_commands
from scripts.serve_edgellm import serve_prebuilt_engines


class DeploymentTests(unittest.TestCase):
    def test_vlm_engine_builder_prepares_llm_then_visual_build(self) -> None:
        edge_root = Path("/opt/TensorRT-Edge-LLM")
        onnx_root = Path("/work/onnx")
        engine_root = Path("/work/engines")

        llm_command, visual_command = build_commands(
            edge_root=edge_root,
            onnx_root=onnx_root,
            engine_root=engine_root,
            max_batch_size=1,
            max_input_len=1024,
            max_kv_cache_capacity=2048,
            min_image_tokens=8,
            max_image_tokens=2048,
            max_image_tokens_per_image=2048,
        )

        self.assertEqual(
            Path(llm_command[0]),
            edge_root / "build" / "examples" / "llm" / "llm_build",
        )
        self.assertIn(str(onnx_root / "llm"), llm_command)
        self.assertIn(str(engine_root / "llm"), llm_command)
        self.assertEqual(
            Path(visual_command[0]),
            edge_root
            / "build"
            / "examples"
            / "multimodal"
            / "visual_build",
        )
        self.assertIn(str(onnx_root / "visual"), visual_command)
        visual_engine_dir_index = visual_command.index("--engineDir") + 1
        self.assertEqual(visual_command[visual_engine_dir_index], str(engine_root))

    def test_server_loads_prebuilt_llm_and_visual_engines(self) -> None:
        calls: dict[str, object] = {}

        class FakeLlm:
            def __init__(self, **kwargs: object) -> None:
                calls["init"] = kwargs

            def serve(self, *, host: str, port: int) -> None:
                calls["serve"] = {"host": host, "port": port}

        experimental_module = ModuleType("experimental")
        server_module = ModuleType("experimental.server")
        server_module.LLM = FakeLlm  # type: ignore[attr-defined]
        experimental_module.server = server_module  # type: ignore[attr-defined]

        with patch.dict(
            "sys.modules",
            {
                "experimental": experimental_module,
                "experimental.server": server_module,
            },
        ):
            serve_prebuilt_engines(
                engine_root=Path("/work/engines"),
                host="127.0.0.1",
                port=8000,
            )

        self.assertEqual(
            calls["init"],
            {
                "engine_dir": str(Path("/work/engines/llm")),
                "visual_engine_dir": str(Path("/work/engines/visual")),
            },
        )
        self.assertEqual(calls["serve"], {"host": "127.0.0.1", "port": 8000})


if __name__ == "__main__":
    unittest.main()
