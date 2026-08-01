"""执行一项已配置研究并写入完整证据报告。"""

from __future__ import annotations

import argparse

from parksight_vlm.casebook import ParkingCaseCatalog
from parksight_vlm.studies import StudyReport, StudyRunner

from .config import AppStudyConfig
from .environment import capture_environment
from .runtime_factory import build_runtime


def run_configured_study(config: AppStudyConfig) -> StudyReport:
    """根据已校验配置组合样本目录、Runtime 和 Runner。"""
    casebook = ParkingCaseCatalog.load(
        config.manifest_path,
        config.annotations_path,
    )
    runtime = build_runtime(config.runtime, data_root=config.data_root)
    report = StudyRunner(environment_provider=capture_environment).run(
        casebook,
        runtime,
        config.study,
    )
    report.write_json(config.output_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = AppStudyConfig.load(args.config)
    report = run_configured_study(config)
    print(config.output_path)
    return 0 if not report.failure_summary else 2


if __name__ == "__main__":
    raise SystemExit(main())
