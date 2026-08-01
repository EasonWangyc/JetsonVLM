"""显式外部模型流程入口共用的 CLI 行为。"""

from __future__ import annotations

import argparse
import json

from parksight_vlm.flows import ExternalFlowPlan, FlowValidationError


def run_stage_cli(expected_stage: str, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run the reviewed command; without this flag only readiness is printed",
    )
    args = parser.parse_args(argv)
    plan = ExternalFlowPlan.load(args.config)
    if plan.stage != expected_stage:
        raise FlowValidationError(
            f"expected stage {expected_stage!r}, got {plan.stage!r}"
        )
    if not args.execute:
        print(json.dumps(plan.readiness_mapping(), ensure_ascii=False, indent=2))
        return 0
    result = plan.execute()
    print(json.dumps(result.to_mapping(), ensure_ascii=False, indent=2))
    return 0 if result.status == "succeeded" else 2
