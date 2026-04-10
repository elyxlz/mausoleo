from __future__ import annotations

import base64
import dataclasses as dc
import io
import json
import typing as tp

from mausoleo.ocr.operators.base import BaseOperatorConfig, OperatorType, register_operator


@dc.dataclass(frozen=True, kw_only=True)
class ColumnSplit(BaseOperatorConfig):
    num_columns: int = 4
    overlap_pct: float = 0.05
    header_crop_pct: float = 0.03
    footer_crop_pct: float = 0.02


@register_operator(ColumnSplit, operation=OperatorType.MAP)
def column_split(row: dict[str, tp.Any], *, config: ColumnSplit) -> dict[str, tp.Any]:
    if config.mock:
        return row

    from PIL import Image

    images_b64 = str(row["images_b64"])
    raw_images = [base64.b64decode(b64) for b64 in images_b64.split("|")]

    all_crops: list[bytes] = []
    all_regions: list[dict[str, tp.Any]] = []

    for page_num, img_bytes in enumerate(raw_images):
        img = Image.open(io.BytesIO(img_bytes))
        w, h = img.size

        top = int(h * config.header_crop_pct)
        bottom = int(h * (1 - config.footer_crop_pct))
        col_width = w / config.num_columns
        overlap = int(col_width * config.overlap_pct)

        for col in range(config.num_columns):
            x1 = max(0, int(col * col_width) - overlap)
            x2 = min(w, int((col + 1) * col_width) + overlap)

            crop = img.crop((x1, top, x2, bottom))
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=95)
            all_crops.append(buf.getvalue())
            all_regions.append({"page": page_num + 1, "column": col + 1, "bbox": [x1, top, x2, bottom]})

    new_b64 = "|".join(base64.b64encode(c).decode() for c in all_crops)
    return {**row, "images_b64": new_b64, "page_count": len(all_crops), "layout_json": json.dumps(all_regions)}
