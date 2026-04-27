from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="gemma3_2b_v1_structured",
    operators=[
        VlmOcr(model="google/gemma-3-4b-it", prompt=prompts.VLM_OCR_STRUCTURED, backend="transformers", max_tokens=4096, max_model_len=16384),
        MergePages(),
        ParseIssue(),
    ],
)
