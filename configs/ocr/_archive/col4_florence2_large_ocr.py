from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergePages, ParseIssue, VlmOcr

LEGACY_ENV = {"pip": ["transformers==4.44.0", "accelerate>=0.25.0", "flash_attn"]}

config = OcrPipelineConfig(
    name="col4_florence2_large_ocr",
    operators=[
        ColumnSplit(num_columns=4, overlap_pct=0.03),
        VlmOcr(model="microsoft/Florence-2-large", prompt="<OCR>", backend="transformers", max_tokens=4096, runtime_env=LEGACY_ENV),
        MergePages(),
        ParseIssue(),
    ],
)
