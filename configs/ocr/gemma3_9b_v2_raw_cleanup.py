from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import LlmCleanup, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="gemma3_9b_v2_raw_cleanup",
    operators=[
        VlmOcr(model="google/gemma-3-9b-vision", prompt=prompts.VLM_OCR_RAW_V2, backend="transformers", max_tokens=8192, max_model_len=16384),
        LlmCleanup(model="Qwen/Qwen2.5-3B-Instruct", prompt=prompts.LLM_CLEANUP_V2, backend="vllm", max_tokens=8192, max_model_len=16384),
        ParseIssue(),
    ],
)
