from __future__ import annotations

import dataclasses as dc

from mausoleo.ocr.operators.base import BaseOperatorConfig


@dc.dataclass(frozen=True, kw_only=True)
class OcrPipelineConfig:
    name: str
    operators: list[BaseOperatorConfig]
