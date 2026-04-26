from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="exp_100_col7_qwen3vl_vllm",
    operators=[
        ColumnSplit(num_columns=7, overlap_pct=0.03),
        VlmOcr(
            model="Qwen/Qwen3-VL-8B-Instruct",
            prompt=prompts.VLM_OCR_STRUCTURED_V2,
            backend="vllm",
            max_tokens=8192,
            max_model_len=8192,
            gpu_fraction=1.0,
        ),
        MergePages(),
        ParseIssue(),
    ],
)
