from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="exp_051_qwen3vl_8b_thinking",
    operators=[
        ColumnSplit(num_columns=3, overlap_pct=0.03),
        VlmOcr(
            model="Qwen/Qwen3-VL-8B-Thinking",
            prompt=prompts.VLM_OCR_STRUCTURED_V2,
            backend="vllm",
            max_tokens=12288,
            max_model_len=16384,
            gpu_fraction=1.0,
        ),
        MergePages(),
        ParseIssue(),
    ],
)
