from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="llama32_11b_v1_structured",
    operators=[
        VlmOcr(model="unsloth/Llama-3.2-11B-Vision-Instruct", prompt=prompts.VLM_OCR_STRUCTURED, backend="transformers", max_tokens=4096),
        MergePages(),
        ParseIssue(),
    ],
)
