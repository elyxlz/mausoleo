from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="exp_049_qwen3vl_30b_moe_awq",
    operators=[
        ColumnSplit(num_columns=3, overlap_pct=0.03),
        VlmOcr(
            model="jart25/Qwen3-VL-30B-A3B-Instruct-AWQ-4bit",
            prompt=prompts.VLM_OCR_STRUCTURED_V2,
            backend="vllm",
            max_tokens=4096,
            max_model_len=8192,
            gpu_fraction=1.0,
        ),
        MergePages(),
        ParseIssue(),
    ],
)
