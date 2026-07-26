"""StudyRunner orchestration from catalog cases to one report."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

from parksight_vlm.casebook import ParkingCaseCatalog
from parksight_vlm.inference import RiskRuntime

from .model import StudyDefinition, StudyReport, StudyValidationError
from .performance import compute_performance_metrics
from .quality import compute_quality_metrics


class StudyRunner:
    """Run a frozen study without depending on a concrete inference backend."""

    def __init__(
        self,
        environment_provider: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self._environment_provider = environment_provider or (lambda: {})

    def run(
        self,
        casebook: ParkingCaseCatalog,
        runtime: RiskRuntime,
        study: StudyDefinition,
    ) -> StudyReport:
        selected_cases = casebook.cases_in_split(study.split)
        if not selected_cases:
            raise StudyValidationError(
                f"study split contains no cases: {study.split.value!r}"
            )

        unlabeled_case_ids = [
            parking_case.case_id
            for parking_case in selected_cases
            if parking_case.reference_assessment is None
        ]
        if unlabeled_case_ids:
            raise StudyValidationError(
                f"study cases require reference assessments: {unlabeled_case_ids}"
            )
        records = tuple(
            runtime.analyze(parking_case, study.workload)
            for parking_case in selected_cases
            for _ in range(study.repetitions)
        )
        references = {
            parking_case.case_id: parking_case.reference_assessment
            for parking_case in selected_cases
            if parking_case.reference_assessment is not None
        }
        failure_summary = Counter(
            record.failure.category.value
            for record in records
            if record.failure is not None
        )
        return StudyReport(
            study_identity=study.identity_mapping(runtime.identity),
            environment_snapshot=dict(self._environment_provider()),
            quality_metrics=compute_quality_metrics(records, references),
            performance_metrics=compute_performance_metrics(records),
            failure_summary=dict(sorted(failure_summary.items())),
            records=records,
        )
