from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="col4_florence2_large_ocr",
    operators=[
        ColumnSplit(num_columns=4, overlap_pct=0.03),
        VlmOcr(model="microsoft/Florence-2-large", prompt="<OCR>", backend="transformers", max_tokens=4096),
        MergePages(),
        ParseIssue(),
    ],
)
