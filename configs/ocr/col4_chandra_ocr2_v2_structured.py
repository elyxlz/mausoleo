from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergePages, ParseIssue, Preprocess, VlmOcr

config = OcrPipelineConfig(
    name="col4_chandra_ocr2_v2_structured",
    operators=[
        Preprocess(grayscale=False, max_dimension=768),
        ColumnSplit(num_columns=4, overlap_pct=0.03),
        VlmOcr(model="datalab-to/chandra-ocr-2", prompt=prompts.VLM_OCR_STRUCTURED_V2, backend="transformers", max_tokens=8192),
        MergePages(),
        ParseIssue(),
    ],
)
