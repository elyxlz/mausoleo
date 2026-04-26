from __future__ import annotations

import base64
import dataclasses as dc
import io
import json
import typing as tp

from mausoleo.ocr.operators.base import BaseOperatorConfig, OperatorType, StatefulOperator, register_operator


@dc.dataclass(frozen=True, kw_only=True)
class YoloCrop(BaseOperatorConfig):
    model: str = "juliozhao/DocLayout-YOLO-DocStructBench"
    conf_threshold: float = 0.25
    text_classes: tuple[str, ...] = ("plain text", "title", "abandon", "figure caption", "table caption")
    gpu_fraction: float = 0.3
    min_region_area: int = 5000
    merge_vertical_gap: int = 50
    merge_horizontal_overlap: float = 0.5
    padding: int = 10


def _merge_column_boxes(
    boxes: list[tuple[int, int, int, int, str, float]],
    vertical_gap: int,
    horizontal_overlap: float,
) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda b: (b[0], b[1]))

    columns: list[list[tuple[int, int, int, int]]] = []
    for x1, y1, x2, y2, _, _ in sorted_boxes:
        box_w = x2 - x1
        best_col = -1
        best_overlap = 0.0

        for ci, col in enumerate(columns):
            col_x1 = min(b[0] for b in col)
            col_x2 = max(b[2] for b in col)
            col_y2 = max(b[3] for b in col)
            col_w = col_x2 - col_x1

            overlap_x = min(x2, col_x2) - max(x1, col_x1)
            if overlap_x <= 0:
                continue

            overlap_ratio = overlap_x / min(box_w, col_w)
            close_enough_y = y1 - col_y2 < vertical_gap

            if overlap_ratio >= horizontal_overlap and close_enough_y and overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_col = ci

        if best_col >= 0:
            columns[best_col].append((x1, y1, x2, y2))
        else:
            columns.append([(x1, y1, x2, y2)])

    merged_boxes = []
    for col in columns:
        mx1 = min(b[0] for b in col)
        my1 = min(b[1] for b in col)
        mx2 = max(b[2] for b in col)
        my2 = max(b[3] for b in col)
        merged_boxes.append((mx1, my1, mx2, my2))

    return sorted(merged_boxes, key=lambda b: (b[0], b[1]))


@register_operator(YoloCrop, operation=OperatorType.MAP_BATCHES)
class YoloCropOperator(StatefulOperator[YoloCrop]):
    def __init__(self, config: YoloCrop) -> None:
        self.config = config
        if config.mock:
            return

        import os, torch

        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        cudnn_lib = os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "cudnn", "lib")
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{torch_lib}:{cudnn_lib}:{existing_ld}"

        try:
            _ = torch.nn.functional.conv2d(
                torch.randn(1, 1, 4, 4, device="cuda"),
                torch.randn(1, 1, 3, 3, device="cuda"),
            )
        except Exception:
            torch.backends.cudnn.enabled = False

        from huggingface_hub import hf_hub_download

        weights_path = hf_hub_download(config.model, "doclayout_yolo_docstructbench_imgsz1024.pt")
        try:
            from doclayout_yolo import YOLOv10
            self.yolo = YOLOv10(weights_path)
        except ImportError:
            from ultralytics import YOLO
            self.yolo = YOLO(weights_path)

    def __call__(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        if self.config.mock:
            return self._mock_call(batch)

        from PIL import Image

        images_b64 = str(batch["images_b64"][0])
        raw_images = [base64.b64decode(b64) for b64 in images_b64.split("|")]

        all_crops: list[bytes] = []
        all_regions: list[dict[str, tp.Any]] = []

        for page_num, img_bytes in enumerate(raw_images):
            pil_img = Image.open(io.BytesIO(img_bytes))
            img_w, img_h = pil_img.size
            results = self.yolo(pil_img, conf=self.config.conf_threshold, verbose=False)

            raw_boxes: list[tuple[int, int, int, int, str, float]] = []
            for box in results[0].boxes:
                cls_name = results[0].names[int(box.cls)]
                if cls_name not in self.config.text_classes:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                area = (x2 - x1) * (y2 - y1)
                if area < self.config.min_region_area:
                    continue
                raw_boxes.append((x1, y1, x2, y2, cls_name, float(box.conf)))

            merged = _merge_column_boxes(raw_boxes, self.config.merge_vertical_gap, self.config.merge_horizontal_overlap)

            if not merged:
                merged = [(0, 0, img_w, img_h)]

            pad = self.config.padding
            for x1, y1, x2, y2 in merged:
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(img_w, x2 + pad)
                y2 = min(img_h, y2 + pad)

                crop = pil_img.crop((x1, y1, x2, y2))
                buf = io.BytesIO()
                crop.save(buf, format="JPEG", quality=95)
                all_crops.append(buf.getvalue())
                all_regions.append({"page": page_num + 1, "bbox": [x1, y1, x2, y2]})

        if not all_crops:
            all_crops = raw_images
            all_regions = [{"page": i + 1, "bbox": [0, 0, 0, 0]} for i in range(len(raw_images))]

        new_b64 = "|".join(base64.b64encode(c).decode() for c in all_crops)
        result = dict(batch)
        result["images_b64"] = [new_b64]
        result["page_count"] = [len(all_crops)]
        result["layout_json"] = [json.dumps(all_regions)]
        return result

    def _mock_call(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        images_b64 = str(batch["images_b64"][0])
        page_count = len(images_b64.split("|"))
        mock_regions = []
        for i in range(page_count):
            mock_regions.extend([
                {"page": i + 1, "bbox": [10, 10, 500, 1500]},
                {"page": i + 1, "bbox": [510, 10, 1000, 1500]},
            ])
        result = dict(batch)
        result["layout_json"] = [json.dumps(mock_regions)]
        return result
