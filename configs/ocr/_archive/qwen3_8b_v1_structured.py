from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="qwen3_8b_v1_structured",
    operators=[
        VlmOcr(model="Qwen/Qwen3-VL-8B-Instruct", prompt=prompts.VLM_OCR_STRUCTURED, backend="transformers", max_tokens=4096, max_model_len=16384),
        MergePages(),
        ParseIssue(),
    ],
)
