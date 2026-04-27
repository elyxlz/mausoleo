from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="exp_131_col10_ads_vllm",
    operators=[
        ColumnSplit(num_columns=10, overlap_pct=0.03),
        VlmOcr(
            model="Qwen/Qwen3-VL-8B-Instruct",
            prompt=prompts.VLM_OCR_ADS_FOCUSED,
            backend="vllm",
            max_tokens=2048,
            max_model_len=8192,
            gpu_fraction=1.0,
        ),
        MergePages(),
        ParseIssue(),
    ],
)
