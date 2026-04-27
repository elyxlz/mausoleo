from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="exp_062_col3_t02",
    operators=[
        ColumnSplit(num_columns=3, overlap_pct=0.03),
        VlmOcr(
            model="Qwen/Qwen3-VL-8B-Instruct",
            prompt=prompts.VLM_OCR_STRUCTURED_V2,
            backend="vllm",
            max_tokens=8192,
            max_model_len=12288,
            gpu_fraction=1.0,
            temperature=0.2,
        ),
        MergePages(),
        ParseIssue(),
    ],
)
