from __future__ import annotations

import base64
import dataclasses as dc
import io
import typing as tp

from mausoleo.ocr.operators.base import BaseOperatorConfig, OperatorType, register_operator


@dc.dataclass(frozen=True, kw_only=True)
class Preprocess(BaseOperatorConfig):
    grayscale: bool = True
    max_dimension: int = 2000


@register_operator(Preprocess, operation=OperatorType.MAP)
def preprocess(row: dict[str, tp.Any], *, config: Preprocess) -> dict[str, tp.Any]:
    if config.mock:
        return row

    from PIL import Image

    images_b64 = str(row["images_b64"])
    raw_images = [base64.b64decode(b64) for b64 in images_b64.split("|")]

    processed: list[bytes] = []
    for img_bytes in raw_images:
        img = Image.open(io.BytesIO(img_bytes))

        if config.grayscale:
            img = img.convert("L")

        max_dim = max(img.size)
        if max_dim > config.max_dimension:
            ratio = config.max_dimension / max_dim
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        processed.append(buf.getvalue())

    new_b64 = "|".join(base64.b64encode(img).decode() for img in processed)
    return {**row, "images_b64": new_b64}
