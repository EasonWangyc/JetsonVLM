"""TensorRT Edge-LLM 部署入口的无硬件行为测试。"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from scripts.build_edgellm_vlm_engines import build_commands
from scripts.serve_edgellm import (
    configure_edge_llm_environment,
    configure_weight_streaming_budget,
    serve_prebuilt_engines,
)


class DeploymentTests(unittest.TestCase):
    def test_edge_llm_environment_discovers_import_and_plugin_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pybind = root / "build" / "pybind"
            pybind.mkdir(parents=True)
            plugin = root / "build" / "libNvInfer_edgellm_plugin.so"
            plugin.write_bytes(b"plugin")
            with patch.dict("os.environ", {}, clear=True), patch.object(
                sys, "path", []
            ):
                configure_edge_llm_environment(root, None)

                self.assertIn(str(root), sys.path)
                self.assertIn(str(pybind), sys.path)
                self.assertEqual(os.environ["BUILD_DIR"], str(root / "build"))
                self.assertEqual(os.environ["EDGELLM_PLUGIN_PATH"], str(plugin))

    def test_edge_llm_environment_rejects_missing_plugin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(FileNotFoundError, "plugin not found"):
                configure_edge_llm_environment(
                    Path(temporary_directory),
                    Path(temporary_directory) / "missing.so",
                )

    def test_edge_llm_environment_rejects_missing_plugin_from_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.dict(
                "os.environ",
                {
                    "EDGELLM_PLUGIN_PATH": str(
                        Path(temporary_directory) / "missing.so"
                    )
                },
                clear=True,
            ):
                with self.assertRaisesRegex(FileNotFoundError, "plugin not found"):
                    configure_edge_llm_environment(None, None)

    def test_weight_streaming_budget_is_exported_for_runtime(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            configure_weight_streaming_budget(0)
            self.assertEqual(
                os.environ["EDGELLM_WEIGHT_STREAMING_BUDGET_BYTES"], "0"
            )

    def test_weight_streaming_budget_rejects_negative_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be negative"):
            configure_weight_streaming_budget(-1)

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
