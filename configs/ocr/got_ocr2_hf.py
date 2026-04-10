from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="got_ocr2_hf",
    operators=[
        VlmOcr(model="stepfun-ai/GOT-OCR-2.0-hf", prompt="", backend="transformers", max_tokens=4096),
        MergePages(),
        ParseIssue(),
    ],
)
