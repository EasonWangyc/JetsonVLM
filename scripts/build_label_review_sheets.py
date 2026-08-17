"""将待复核图片生成带样本名的联系表，便于逐图人工标注。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not records:
        raise ValueError("review dataset must not be empty")
    return records


def _resolve_image(record: dict[str, Any], image_root: Path) -> Path:
    image_reference = record.get("image") or record.get("image_ref")
    if image_reference is None:
        raise ValueError("review record requires image or image_ref")
    image_name = Path(str(image_reference)).name
    image_path = image_root / image_name
    if not image_path.is_file():
        raise FileNotFoundError(f"missing review image: {image_path}")
    return image_path


def build_contact_sheets(
    *,
    records: list[dict[str, Any]],
    image_root: Path,
    output_directory: Path,
    columns: int,
    rows: int,
    cell_width: int,
    cell_height: int,
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
    header_height = 30
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
            label = f"{page_index * page_size + cell_index + 1:02d}  {image_path.name}"
            draw.text((left + 6, top + 5), label, fill="black", font=font)
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
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cell-width", type=int, default=360)
    parser.add_argument("--cell-height", type=int, default=270)
    args = parser.parse_args()

    paths = build_contact_sheets(
        records=_load_records(args.records),
        image_root=args.image_root,
        output_directory=args.output_directory,
        columns=args.columns,
        rows=args.rows,
        cell_width=args.cell_width,
        cell_height=args.cell_height,
    )
    print(json.dumps({"sheets": [str(path) for path in paths]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
