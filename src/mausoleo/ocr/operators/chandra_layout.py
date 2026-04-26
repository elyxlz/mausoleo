from __future__ import annotations

import base64
import dataclasses as dc
import io
import json
import re
import typing as tp

from mausoleo.ocr.operators.base import BaseOperatorConfig, OperatorType, StatefulOperator, register_operator


@dc.dataclass(frozen=True, kw_only=True)
class ChandraLayout(BaseOperatorConfig):
    model: str = "datalab-to/chandra-ocr-2"
    text_labels: tuple[str, ...] = ("Text", "Section-Header", "Title", "Caption", "List-item")
    gpu_fraction: float = 1.0
    min_region_area: int = 3000
    merge_vertical_gap: int = 50
    merge_horizontal_overlap: float = 0.5
    padding: int = 10
    max_tokens: int = 4096


def _parse_chandra_layout(raw_text: str) -> list[dict[str, tp.Any]]:
    raw_text = raw_text.strip()
    first_chunk = raw_text.split("\nassistant\n")[0] if "\nassistant\n" in raw_text else raw_text
    first_chunk = first_chunk.strip()
    if not first_chunk.startswith("["):
        match = re.search(r"\[\s*\{", first_chunk)
        if match:
            first_chunk = first_chunk[match.start():]
    depth = 0
    end_idx = -1
    for i, ch in enumerate(first_chunk):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end_idx = i
                break
    if end_idx < 0:
        return []
    try:
        return json.loads(first_chunk[:end_idx + 1])
    except json.JSONDecodeError:
        return []


def _parse_bbox(bbox_val: tp.Any) -> tuple[int, int, int, int] | None:
    if isinstance(bbox_val, str):
        parts = bbox_val.split()
        if len(parts) != 4:
            return None
        try:
            return tuple(int(p) for p in parts)  # type: ignore[return-value]
        except ValueError:
            return None
    if isinstance(bbox_val, (list, tuple)) and len(bbox_val) == 4:
        try:
            return tuple(int(v) for v in bbox_val)  # type: ignore[return-value]
        except (ValueError, TypeError):
            return None
    return None


def _merge_column_boxes(
    boxes: list[tuple[int, int, int, int]],
    vertical_gap: int,
    horizontal_overlap: float,
) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
    columns: list[list[tuple[int, int, int, int]]] = []
    for x1, y1, x2, y2 in sorted_boxes:
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


@register_operator(ChandraLayout, operation=OperatorType.MAP_BATCHES)
class ChandraLayoutOperator(StatefulOperator[ChandraLayout]):
    def __init__(self, config: ChandraLayout) -> None:
        self.config = config
        if config.mock:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(config.model, trust_remote_code=True)
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        self.hf_model = AutoModelForImageTextToText.from_pretrained(
            config.model, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16
        )

    def _detect_layout(self, pil_img: tp.Any) -> list[dict[str, tp.Any]]:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": "Detect the layout regions in this document. Output JSON array of regions with 'label' and 'bbox' (normalized 0-1000)."},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[pil_img], return_tensors="pt").to(self.hf_model.device)
        with torch.no_grad():
            output_ids = self.hf_model.generate(**inputs, max_new_tokens=self.config.max_tokens, do_sample=False)
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        raw = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return _parse_chandra_layout(raw)

    def __call__(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        if self.config.mock:
            return self._mock_call(batch)

        from PIL import Image

        images_b64 = str(batch["images_b64"][0])
        raw_images = [base64.b64decode(b64) for b64 in images_b64.split("|")]

        all_crops: list[bytes] = []
        all_regions: list[dict[str, tp.Any]] = []

        for page_num, img_bytes in enumerate(raw_images):
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_w, img_h = pil_img.size

            layout = self._detect_layout(pil_img)

            raw_boxes: list[tuple[int, int, int, int]] = []
            for item in layout:
                label = item.get("label", "")
                if label not in self.config.text_labels:
                    continue
                bbox = _parse_bbox(item.get("bbox"))
                if bbox is None:
                    continue
                nx1, ny1, nx2, ny2 = bbox
                x1 = int(nx1 * img_w / 1000)
                y1 = int(ny1 * img_h / 1000)
                x2 = int(nx2 * img_w / 1000)
                y2 = int(ny2 * img_h / 1000)
                area = (x2 - x1) * (y2 - y1)
                if area < self.config.min_region_area:
                    continue
                raw_boxes.append((x1, y1, x2, y2))

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
        mock_regions = [{"page": i + 1, "bbox": [10, 10, 500, 1500]} for i in range(page_count)]
        result = dict(batch)
        result["layout_json"] = [json.dumps(mock_regions)]
        return result
