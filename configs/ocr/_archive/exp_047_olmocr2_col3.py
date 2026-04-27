from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="exp_047_olmocr2_col3",
    operators=[
        ColumnSplit(num_columns=3, overlap_pct=0.03),
        VlmOcr(
            model="allenai/olmOCR-2-7B-1025",
            prompt=prompts.VLM_OCR_STRUCTURED_V2,
            backend="vllm",
            max_tokens=8192,
            max_model_len=16384,
            gpu_fraction=1.0,
        ),
        MergePages(),
        ParseIssue(),
    ],
)
