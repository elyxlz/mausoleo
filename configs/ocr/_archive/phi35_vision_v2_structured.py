from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, Preprocess, VlmOcr

config = OcrPipelineConfig(
    name="phi35_vision_v2_structured",
    operators=[
        Preprocess(grayscale=False, max_dimension=1024),
        VlmOcr(model="microsoft/Phi-3.5-vision-instruct", prompt=prompts.VLM_OCR_STRUCTURED_V2, backend="transformers", max_tokens=8192),
        MergePages(),
        ParseIssue(),
    ],
)
