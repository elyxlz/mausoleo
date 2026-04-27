from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="qwen25_3b_v1_structured",
    operators=[
        VlmOcr(model="Qwen/Qwen2.5-VL-3B-Instruct", prompt=prompts.VLM_OCR_STRUCTURED, backend="vllm", max_tokens=4096, max_model_len=16384),
        MergePages(),
        ParseIssue(),
    ],
)
