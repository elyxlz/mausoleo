from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="col4_got_ocr2_hf",
    operators=[
        ColumnSplit(num_columns=4, overlap_pct=0.03),
        VlmOcr(model="stepfun-ai/GOT-OCR-2.0-hf", prompt="", backend="transformers", max_tokens=4096),
        MergePages(),
        ParseIssue(),
    ],
)
