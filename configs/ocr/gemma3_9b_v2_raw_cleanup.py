from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import LlmCleanup, ParseIssue, Preprocess, VlmOcr

config = OcrPipelineConfig(
    name="gemma3_9b_v2_raw_cleanup",
    operators=[
        Preprocess(grayscale=False, max_dimension=1024),
        VlmOcr(model="google/gemma-3-12b-it", prompt=prompts.VLM_OCR_RAW_V2, backend="transformers", max_tokens=8192, max_model_len=16384, gpu_fraction=2.0),
        LlmCleanup(model="Qwen/Qwen2.5-3B-Instruct", prompt=prompts.LLM_CLEANUP_V2, backend="vllm", max_tokens=8192, max_model_len=16384),
        ParseIssue(),
    ],
)
