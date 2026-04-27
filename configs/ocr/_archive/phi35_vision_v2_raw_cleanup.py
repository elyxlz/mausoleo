from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import LlmCleanup, ParseIssue, Preprocess, VlmOcr

config = OcrPipelineConfig(
    name="phi35_vision_v2_raw_cleanup",
    operators=[
        Preprocess(grayscale=False, max_dimension=1024),
        VlmOcr(model="microsoft/Phi-3.5-vision-instruct", prompt=prompts.VLM_OCR_RAW_V2, backend="transformers", max_tokens=8192),
        LlmCleanup(model="Qwen/Qwen2.5-3B-Instruct", prompt=prompts.LLM_CLEANUP_V2, backend="vllm", max_tokens=8192, max_model_len=16384),
        ParseIssue(),
    ],
)
