from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, LlmCleanup, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="exp_009_col3_raw_cleanup",
    operators=[
        ColumnSplit(num_columns=3, overlap_pct=0.03),
        VlmOcr(model="Qwen/Qwen3-VL-8B-Instruct", prompt=prompts.VLM_OCR_COLUMN, backend="transformers", max_tokens=8192, max_model_len=32768),
        LlmCleanup(model="Qwen/Qwen2.5-3B-Instruct", prompt=prompts.LLM_CLEANUP_V2, backend="vllm", max_tokens=16384, max_model_len=32768),
        ParseIssue(),
    ],
)
