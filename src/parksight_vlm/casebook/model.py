"""Domain objects for annotated parking-scene samples."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath

from parksight_vlm.assessment import ParkingAssessment


class CasebookValidationError(ValueError):
    """Raised when case metadata or a catalog invariant is invalid."""


class DatasetSplit(str, Enum):
    """Dataset partitions used by quality studies."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class ParkingCase:
    """One parking image together with its source group and reference label."""

    case_id: str
    image_ref: PurePosixPath
    source_group_id: str
    split: DatasetSplit
    reference_assessment: ParkingAssessment | None

    def resolve_image(self, data_root: Path, *, require_exists: bool = False) -> Path:
        """Resolve an image reference below ``data_root`` without escaping it."""
        root = data_root.resolve()
        image_path = (root / Path(*self.image_ref.parts)).resolve()
        try:
            image_path.relative_to(root)
        except ValueError as error:
            raise CasebookValidationError(
                f"image_ref escapes data_root: {self.image_ref.as_posix()!r}"
            ) from error
        if require_exists and not image_path.is_file():
            raise CasebookValidationError(f"image file does not exist: {image_path}")
        return image_path
