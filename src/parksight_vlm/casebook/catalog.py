"""泊车样本目录的 JSONL 加载与不变量校验。

Manifest 行必须且只能使用 ``case_id``、``image_ref``、``source_group_id`` 和
``split``。Annotation 行必须且只能使用 ``case_id`` 和 ``assessment``，其中
``assessment`` 遵循 :class:`ParkingAssessment` 的严格 JSON 契约。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from parksight_vlm.assessment import AssessmentValidationError, ParkingAssessment

from .model import CasebookValidationError, DatasetSplit, ParkingCase


@dataclass(frozen=True, slots=True)
class ParkingCaseCatalog:
    """从一份 manifest 和一份 annotation 文件加载并校验的泊车样本。"""

    cases: tuple[ParkingCase, ...]
    manifest_path: Path
    annotations_path: Path

    @classmethod
    def load(
        cls,
        manifest_path: Path | str,
        annotations_path: Path | str,
    ) -> "ParkingCaseCatalog":
        """加载相互匹配的 JSONL 元数据和参考评估记录。"""
        resolved_manifest_path = Path(manifest_path)
        resolved_annotations_path = Path(annotations_path)
        manifest_records = _read_jsonl(resolved_manifest_path)
        annotation_records = _read_jsonl(resolved_annotations_path)

        manifests_by_id = _index_manifest_records(manifest_records, resolved_manifest_path)
        assessments_by_id = _index_annotation_records(
            annotation_records, resolved_annotations_path
        )
        manifest_ids = set(manifests_by_id)
        annotation_ids = set(assessments_by_id)
        missing_annotations = manifest_ids - annotation_ids
        unexpected_annotations = annotation_ids - manifest_ids
        if missing_annotations or unexpected_annotations:
            details = []
            if missing_annotations:
                details.append(
                    "missing annotations for case_ids: "
                    f"{sorted(missing_annotations)}"
                )
            if unexpected_annotations:
                details.append(
                    "annotations without a manifest: "
                    f"{sorted(unexpected_annotations)}"
                )
            raise CasebookValidationError("; ".join(details))

        cases = tuple(
            _build_case(manifest, assessments_by_id[case_id])
            for case_id, manifest in manifests_by_id.items()
        )
        catalog = cls(
            cases=cases,
            manifest_path=resolved_manifest_path,
            annotations_path=resolved_annotations_path,
        )
        catalog.validate()
        return catalog

    def validate(self) -> None:
        """校验目录范围内的身份与来源组划分不变量。"""
        if not self.cases:
            raise CasebookValidationError("case catalog must contain at least one case")

        case_ids: set[str] = set()
        split_by_source_group: dict[str, DatasetSplit] = {}
        for parking_case in self.cases:
            if parking_case.case_id in case_ids:
                raise CasebookValidationError(
                    f"duplicate case_id: {parking_case.case_id!r}"
                )
            case_ids.add(parking_case.case_id)

            existing_split = split_by_source_group.get(parking_case.source_group_id)
            if existing_split is None:
                split_by_source_group[parking_case.source_group_id] = parking_case.split
            elif existing_split != parking_case.split:
                raise CasebookValidationError(
                    "source_group_id appears in multiple splits: "
                    f"{parking_case.source_group_id!r} is both "
                    f"{existing_split.value!r} and {parking_case.split.value!r}"
                )

    def cases_in_split(self, split: DatasetSplit) -> tuple[ParkingCase, ...]:
        """返回目录中属于指定数据集划分的样本。"""
        return tuple(parking_case for parking_case in self.cases if parking_case.split == split)


def _read_jsonl(path: Path) -> list[tuple[int, Mapping[str, Any]]]:
    try:
        contents = path.read_text(encoding="utf-8")
    except OSError as error:
        raise CasebookValidationError(f"cannot read JSONL file: {path}") from error

    records: list[tuple[int, Mapping[str, Any]]] = []
    for line_number, line in enumerate(contents.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise CasebookValidationError(
                f"invalid JSON at {path}:{line_number}"
            ) from error
        if not isinstance(record, Mapping):
            raise CasebookValidationError(
                f"JSONL record must be an object at {path}:{line_number}"
            )
        records.append((line_number, record))
    return records


def _index_manifest_records(
    records: Iterable[tuple[int, Mapping[str, Any]]], path: Path
) -> dict[str, Mapping[str, Any]]:
    manifests_by_id: dict[str, Mapping[str, Any]] = {}
    for line_number, record in records:
        _require_exact_fields(
            record,
            {"case_id", "image_ref", "source_group_id", "split"},
            path,
            line_number,
        )
        case_id = _parse_identifier(record["case_id"], "case_id", path, line_number)
        if case_id in manifests_by_id:
            raise CasebookValidationError(
                f"duplicate case_id {case_id!r} at {path}:{line_number}"
            )
        manifests_by_id[case_id] = record
    return manifests_by_id


def _index_annotation_records(
    records: Iterable[tuple[int, Mapping[str, Any]]], path: Path
) -> dict[str, ParkingAssessment]:
    assessments_by_id: dict[str, ParkingAssessment] = {}
    for line_number, record in records:
        _require_exact_fields(record, {"case_id", "assessment"}, path, line_number)
        case_id = _parse_identifier(record["case_id"], "case_id", path, line_number)
        if case_id in assessments_by_id:
            raise CasebookValidationError(
                f"duplicate annotation case_id {case_id!r} at {path}:{line_number}"
            )
        try:
            assessment = ParkingAssessment.from_mapping(record["assessment"])
        except AssessmentValidationError as error:
            raise CasebookValidationError(
                f"invalid assessment at {path}:{line_number}: {error}"
            ) from error
        assessments_by_id[case_id] = assessment
    return assessments_by_id


def _build_case(
    manifest: Mapping[str, Any], reference_assessment: ParkingAssessment
) -> ParkingCase:
    case_id = _parse_identifier(manifest["case_id"], "case_id")
    source_group_id = _parse_identifier(manifest["source_group_id"], "source_group_id")
    image_ref = _parse_image_ref(manifest["image_ref"])
    split = _parse_split(manifest["split"])
    return ParkingCase(
        case_id=case_id,
        image_ref=image_ref,
        source_group_id=source_group_id,
        split=split,
        reference_assessment=reference_assessment,
    )


def _require_exact_fields(
    record: Mapping[str, Any], expected_fields: set[str], path: Path, line_number: int
) -> None:
    actual_fields = set(record)
    missing_fields = expected_fields - actual_fields
    unexpected_fields = actual_fields - expected_fields
    if missing_fields or unexpected_fields:
        details = []
        if missing_fields:
            details.append(f"missing fields: {sorted(missing_fields)}")
        if unexpected_fields:
            details.append(f"unexpected fields: {sorted(unexpected_fields)}")
        raise CasebookValidationError(f"invalid record at {path}:{line_number}; " + "; ".join(details))


def _parse_identifier(
    value: Any,
    field_name: str,
    path: Path | None = None,
    line_number: int | None = None,
) -> str:
    if not isinstance(value, str) or not value.strip():
        location = ""
        if path is not None and line_number is not None:
            location = f" at {path}:{line_number}"
        raise CasebookValidationError(f"{field_name} must be a non-blank string{location}")
    return value.strip()


def _parse_image_ref(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise CasebookValidationError("image_ref must be a non-blank string")
    if "\\" in value:
        raise CasebookValidationError("image_ref must use POSIX separators")
    image_ref = PurePosixPath(value)
    if image_ref.is_absolute() or ".." in image_ref.parts:
        raise CasebookValidationError("image_ref must stay below the data root")
    if image_ref == PurePosixPath("."):
        raise CasebookValidationError("image_ref must name an image file")
    return image_ref


def _parse_split(value: Any) -> DatasetSplit:
    if not isinstance(value, str):
        raise CasebookValidationError("split must be a string")
    try:
        return DatasetSplit(value)
    except ValueError as error:
        allowed_values = ", ".join(split.value for split in DatasetSplit)
        raise CasebookValidationError(
            f"split must be one of: {allowed_values}"
        ) from error
