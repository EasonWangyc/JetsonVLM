"""Validate or explicitly execute the TensorRT Edge-LLM export flow."""

from _flow_cli import run_stage_cli


if __name__ == "__main__":
    raise SystemExit(run_stage_cli("export_model"))
