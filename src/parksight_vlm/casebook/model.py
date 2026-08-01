"""带标注泊车场景样本的领域对象。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath

from parksight_vlm.assessment import ParkingAssessment


class CasebookValidationError(ValueError):
    """当样本元数据或目录不变量无效时抛出。"""


class DatasetSplit(str, Enum):
    """质量研究使用的数据集划分。"""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class ParkingCase:
    """一张泊车图片及其来源组和参考标注。"""

    case_id: str
    image_ref: PurePosixPath
    source_group_id: str
    split: DatasetSplit
    reference_assessment: ParkingAssessment | None

    def resolve_image(self, data_root: Path, *, require_exists: bool = False) -> Path:
        """在不越出 ``data_root`` 的前提下解析图片引用。"""
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
