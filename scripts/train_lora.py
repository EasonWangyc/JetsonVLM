"""Validate or explicitly execute the domain LoRA training flow."""

from _flow_cli import run_stage_cli


if __name__ == "__main__":
    raise SystemExit(run_stage_cli("train_lora"))
