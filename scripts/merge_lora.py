"""Validate or explicitly execute the LoRA merge flow."""

from _flow_cli import run_stage_cli


if __name__ == "__main__":
    raise SystemExit(run_stage_cli("merge_lora"))
