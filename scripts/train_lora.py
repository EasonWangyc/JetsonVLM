"""校验或显式执行领域 LoRA 训练流程。"""

from _flow_cli import run_stage_cli


if __name__ == "__main__":
    raise SystemExit(run_stage_cli("train_lora"))
