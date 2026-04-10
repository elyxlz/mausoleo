from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="florence2_large_ocr",
    operators=[
        VlmOcr(model="microsoft/Florence-2-large", prompt="<OCR>", backend="transformers", max_tokens=4096),
        MergePages(),
        ParseIssue(),
    ],
)
