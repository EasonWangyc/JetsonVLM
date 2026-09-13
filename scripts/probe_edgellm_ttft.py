"""Measure client-observed TTFT from one Edge-LLM streaming request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from parksight_vlm.inference.edge_llm import EdgeLlmHttpBackend


def _load_probe_workload(input_path: Path) -> tuple[Path, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    request = payload["requests"][0]
    messages = request["messages"]
    user_content = messages[1]["content"]
    image_path = Path(next(item["image"] for item in user_content if item.get("type") == "image"))
    user_prompt = next(item["text"] for item in user_content if item.get("type") == "text")
    workload = SimpleNamespace(
        system_prompt=messages[0]["content"],
        generation=SimpleNamespace(
            max_new_tokens=payload.get("max_generate_length", 32),
            do_sample=False,
        ),
        render_user_prompt=lambda: user_prompt,
    )
    return image_path, workload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    image_path, workload = _load_probe_workload(args.input)
    generation = EdgeLlmHttpBackend(
        base_url=args.base_url,
        stream_responses=True,
    ).generate(image_path=image_path, workload=workload)
    result = {
        "image_path": str(image_path),
        "output_tokens": generation.output_tokens,
        "stage_timings": generation.stage_timings.to_mapping(),
        "raw_output": generation.raw_output,
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.write_text(serialized, encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
