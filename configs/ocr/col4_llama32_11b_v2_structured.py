from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="col4_llama32_11b_v2_structured",
    operators=[
        ColumnSplit(num_columns=4, overlap_pct=0.03),
        VlmOcr(model="unsloth/Llama-3.2-11B-Vision-Instruct", prompt=prompts.VLM_OCR_STRUCTURED_V2, backend="transformers", max_tokens=8192),
        MergePages(),
        ParseIssue(),
    ],
)
