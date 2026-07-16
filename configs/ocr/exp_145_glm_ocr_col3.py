from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergeMarkdownPages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="exp_145_glm_ocr_col3",
    operators=[
        ColumnSplit(num_columns=3, overlap_pct=0.03),
        VlmOcr(
            model="zai-org/GLM-OCR",
            prompt="Text Recognition:",
            backend="vllm",
            max_tokens=8192,
            max_model_len=32768,
            gpu_fraction=1.0,
        ),
        MergeMarkdownPages(),
        ParseIssue(),
    ],
)
