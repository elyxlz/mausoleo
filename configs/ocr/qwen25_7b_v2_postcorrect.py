from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import LlmPostCorrect, MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="qwen25_7b_v2_postcorrect",
    operators=[
        VlmOcr(model="Qwen/Qwen2.5-VL-7B-Instruct", prompt=prompts.VLM_OCR_STRUCTURED_V2, backend="vllm", max_tokens=8192, max_model_len=16384),
        LlmPostCorrect(model="Qwen/Qwen2.5-3B-Instruct", prompt=prompts.LLM_POST_CORRECTION, backend="vllm", max_tokens=8192, max_model_len=16384),
        MergePages(),
        ParseIssue(),
    ],
)
