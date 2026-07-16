from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergeMarkdownPages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="exp_151_olmocr_native_col3",
    operators=[
        ColumnSplit(num_columns=3, overlap_pct=0.03),
        VlmOcr(
            model="allenai/olmOCR-2-7B-1025",
            prompt=prompts.VLM_OCR_OLMOCR_NATIVE,
            backend="vllm",
            max_tokens=8192,
            max_model_len=16384,
            gpu_fraction=1.0,
        ),
        MergeMarkdownPages(),
        ParseIssue(),
    ],
)
