from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="phi35_vision_v1_structured",
    operators=[
        VlmOcr(model="microsoft/Phi-3.5-vision-instruct", prompt=prompts.VLM_OCR_STRUCTURED, backend="transformers", max_tokens=4096),
        MergePages(),
        ParseIssue(),
    ],
)
