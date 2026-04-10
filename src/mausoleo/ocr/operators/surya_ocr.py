from __future__ import annotations

import base64
import dataclasses as dc
import io
import json
import typing as tp

from mausoleo.ocr.operators.base import BaseOperatorConfig, OperatorType, StatefulOperator, register_operator


@dc.dataclass(frozen=True, kw_only=True)
class SuryaOcr(BaseOperatorConfig):
    langs: tuple[str, ...] = ("it",)
    gpu_fraction: float = 0.5


@register_operator(SuryaOcr, operation=OperatorType.MAP_BATCHES)
class SuryaOcrOperator(StatefulOperator[SuryaOcr]):
    def __init__(self, config: SuryaOcr) -> None:
        self.config = config
        if config.mock:
            return
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor

        self.det_predictor = DetectionPredictor()
        self.rec_predictor = RecognitionPredictor()

    def __call__(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        if self.config.mock:
            return self._mock_call(batch)

        from PIL import Image
        from surya.recognition import run_recognition

        images_b64 = str(batch["images_b64"][0])
        raw_images = [base64.b64decode(b64) for b64 in images_b64.split("|")]
        pil_images = [Image.open(io.BytesIO(img)) for img in raw_images]

        langs = [list(self.config.langs)] * len(pil_images)
        predictions = run_recognition(pil_images, langs, self.det_predictor, self.rec_predictor)

        page_texts: list[str] = []
        for pred in predictions:
            lines = [line.text for line in pred.text_lines]
            page_texts.append("\n".join(lines))

        result = dict(batch)
        result["page_texts"] = [json.dumps(page_texts)]
        return result

    def _mock_call(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        images_b64 = str(batch["images_b64"][0])
        page_count = len(images_b64.split("|"))
        page_texts = [f"Surya OCR pagina {i + 1}.\nColonna sinistra: notizie locali.\nColonna destra: cronaca." for i in range(page_count)]
        result = dict(batch)
        result["page_texts"] = [json.dumps(page_texts)]
        return result
