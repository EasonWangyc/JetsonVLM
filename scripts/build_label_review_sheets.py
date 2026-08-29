"""将待复核图片生成带样本名的联系表，便于逐图人工标注。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not records:
        raise ValueError("review dataset must not be empty")
    return records


def _resolve_image(record: dict[str, Any], image_root: Path) -> Path:
    image_reference = record.get("image") or record.get("image_ref")
    if image_reference is None:
        raise ValueError("review record requires image or image_ref")
    reference_path = Path(str(image_reference))
    direct_candidates = [image_root / reference_path, image_root / reference_path.name]
    for image_path in direct_candidates:
        if image_path.is_file():
            return image_path

    matches = list(image_root.rglob(reference_path.name))
    if not matches:
        raise FileNotFoundError(
            f"missing review image: {image_root / reference_path.name}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous review image {reference_path.name}: "
            f"{[str(path) for path in matches]}"
        )
    return matches[0]


def _load_candidate_annotations(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    annotations: dict[str, dict[str, Any]] = {}
    for record in _load_records(path):
        if set(record) != {"case_id", "assessment"}:
            raise ValueError("candidate annotation must contain case_id and assessment")
        case_id = str(record["case_id"])
        if case_id in annotations:
            raise ValueError(f"duplicate candidate annotation: {case_id}")
        annotations[case_id] = record["assessment"]
    return annotations


def build_contact_sheets(
    *,
    records: list[dict[str, Any]],
    image_root: Path,
    output_directory: Path,
    columns: int,
    rows: int,
    cell_width: int,
    cell_height: int,
    candidate_annotations: Mapping[str, dict[str, Any]] | None = None,
) -> list[Path]:
    """按固定网格生成联系表，返回生成文件列表。"""
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
    except ImportError as error:
        raise RuntimeError("生成审核联系表需要 Pillow") from error

    if columns <= 0 or rows <= 0:
        raise ValueError("columns and rows must be positive")

    output_directory.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=18)
    header_height = 46
    page_size = columns * rows
    page_count = math.ceil(len(records) / page_size)
    written: list[Path] = []

    for page_index in range(page_count):
        page_records = records[page_index * page_size : (page_index + 1) * page_size]
        sheet = Image.new(
            "RGB",
            (columns * cell_width, rows * (cell_height + header_height)),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for cell_index, record in enumerate(page_records):
            row, column = divmod(cell_index, columns)
            left = column * cell_width
            top = row * (cell_height + header_height)
            image_path = _resolve_image(record, image_root)
            with Image.open(image_path) as source:
                preview = ImageOps.contain(
                    source.convert("RGB"),
                    (cell_width, cell_height),
                )
            image_left = left + (cell_width - preview.width) // 2
            image_top = top + header_height + (cell_height - preview.height) // 2
            sheet.paste(preview, (image_left, image_top))
            case_id = str(record.get("case_id", image_path.stem))
            index_label = f"{page_index * page_size + cell_index + 1:02d}  {case_id}"
            draw.text((left + 6, top + 5), index_label, fill="black", font=font)
            if candidate_annotations and case_id in candidate_annotations:
                assessment = candidate_annotations[case_id]
                risk_level = str(assessment.get("risk_level", "unknown"))
                events = ",".join(str(event) for event in assessment.get("events", []))
                candidate_label = f"candidate: {risk_level} | {events or 'no_event'}"
                draw.text((left + 6, top + 23), candidate_label, fill="#444444", font=font)
            draw.rectangle(
                (left, top, left + cell_width - 1, top + cell_height + header_height - 1),
                outline="#777777",
                width=1,
            )

        output_path = output_directory / f"review_sheet_{page_index + 1:02d}.jpg"
        sheet.save(output_path, quality=92)
        written.append(output_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cell-width", type=int, default=360)
    parser.add_argument("--cell-height", type=int, default=270)
    args = parser.parse_args()

    records = _load_records(args.records)
    paths = build_contact_sheets(
        records=records,
        image_root=args.image_root,
        output_directory=args.output_directory,
        columns=args.columns,
        rows=args.rows,
        cell_width=args.cell_width,
        cell_height=args.cell_height,
        candidate_annotations=_load_candidate_annotations(args.annotations),
    )
    print(json.dumps({"sheets": [str(path) for path in paths]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
