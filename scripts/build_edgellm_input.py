"""Build a TensorRT Edge-LLM ``llm_inference`` input from a frozen manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from parksight_vlm.workload import FrozenWorkload


def build_input(
    *,
    manifest_path: Path,
    workload_path: Path,
    data_root: Path,
    split: str,
    limit: int | None,
    max_generate_length: int | None,
) -> dict[str, Any]:
    workload = FrozenWorkload.load(workload_path)
    requests: list[dict[str, Any]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["split"] != split:
            continue
        image_path = (data_root / Path(*Path(record["image_ref"]).parts)).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"manifest image not found: {image_path}")
        requests.append(
            {
                "messages": [
                    {"role": "system", "content": workload.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path)},
                            {"type": "text", "text": workload.render_user_prompt()},
                        ],
                    },
                ]
            }
        )
        if limit is not None and len(requests) >= limit:
            break
    if not requests:
        raise ValueError(f"no manifest records found for split {split!r}")
    return {
        "batch_size": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_generate_length": (
            max_generate_length
            if max_generate_length is not None
            else workload.generation.max_new_tokens
        ),
        "requests": requests,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-generate-length", type=int, default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    payload = build_input(
        manifest_path=args.manifest,
        workload_path=args.workload,
        data_root=args.data_root,
        split=args.split,
        limit=args.limit,
        max_generate_length=args.max_generate_length,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    print(f"requests={len(payload['requests'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
