from __future__ import annotations

import base64
import dataclasses as dc
import io
import json
import typing as tp

from mausoleo.ocr.operators.base import BaseOperatorConfig, OperatorType, StatefulOperator, register_operator


@dc.dataclass(frozen=True, kw_only=True)
class YoloLayout(BaseOperatorConfig):
    model: str = "yolov8n.pt"
    conf_threshold: float = 0.25
    gpu_fraction: float = 0.2


@register_operator(YoloLayout, operation=OperatorType.MAP_BATCHES)
class YoloLayoutOperator(StatefulOperator[YoloLayout]):
    def __init__(self, config: YoloLayout) -> None:
        self.config = config
        if config.mock:
            return
        from ultralytics import YOLO

        self.yolo = YOLO(config.model)

    def __call__(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        if self.config.mock:
            return self._mock_call(batch)

        from PIL import Image

        images_b64 = str(batch["images_b64"][0])
        raw_images = [base64.b64decode(b64) for b64 in images_b64.split("|")]

        all_layouts: list[list[dict[str, tp.Any]]] = []
        for img_bytes in raw_images:
            pil_img = Image.open(io.BytesIO(img_bytes))
            results = self.yolo(pil_img, conf=self.config.conf_threshold, verbose=False)
            regions: list[dict[str, tp.Any]] = []
            for box in results[0].boxes:
                regions.append(
                    {
                        "class": results[0].names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": [float(x) for x in box.xyxy[0].tolist()],
                    }
                )
            regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
            all_layouts.append(regions)

        result = dict(batch)
        result["layout_json"] = [json.dumps(all_layouts)]
        return result

    def _mock_call(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        images_b64 = str(batch["images_b64"][0])
        page_count = len(images_b64.split("|"))
        all_layouts = [
            [
                {"class": "title", "confidence": 0.95, "bbox": [10, 10, 500, 60]},
                {"class": "text", "confidence": 0.90, "bbox": [10, 70, 250, 800]},
                {"class": "text", "confidence": 0.88, "bbox": [260, 70, 500, 800]},
            ]
            for _ in range(page_count)
        ]
        result = dict(batch)
        result["layout_json"] = [json.dumps(all_layouts)]
        return result
