from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr

LEGACY_ENV = {"pip": ["transformers==4.44.0", "accelerate>=0.25.0"]}

config = OcrPipelineConfig(
    name="phi35_vision_v1_structured",
    operators=[
        VlmOcr(model="microsoft/Phi-3.5-vision-instruct", prompt=prompts.VLM_OCR_STRUCTURED, backend="transformers", max_tokens=4096, runtime_env=LEGACY_ENV),
        MergePages(),
        ParseIssue(),
    ],
)
