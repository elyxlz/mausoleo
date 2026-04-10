from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr

LEGACY_ENV = {"pip": ["transformers==4.44.0", "accelerate>=0.25.0"]}

config = OcrPipelineConfig(
    name="florence2_base_ocr",
    operators=[
        VlmOcr(model="microsoft/Florence-2-base", prompt="<OCR>", backend="transformers", max_tokens=4096, runtime_env=LEGACY_ENV),
        MergePages(),
        ParseIssue(),
    ],
)
