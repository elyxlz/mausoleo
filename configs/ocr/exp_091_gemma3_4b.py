from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="exp_091_gemma3_4b",
    operators=[
        ColumnSplit(num_columns=3, overlap_pct=0.03),
        VlmOcr(
            model="google/gemma-3-4b-it",
            prompt=prompts.VLM_OCR_STRUCTURED_V2,
            backend="transformers",
            max_tokens=8192,
            max_model_len=16384,
            gpu_fraction=1.0,
        ),
        MergePages(),
        ParseIssue(),
    ],
)
