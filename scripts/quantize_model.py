"""校验或显式执行 TensorRT Edge-LLM checkpoint 量化流程。"""

from _flow_cli import run_stage_cli


if __name__ == "__main__":
    raise SystemExit(run_stage_cli("quantize_model"))
