from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="exp_007_no_crop",
    operators=[
        ColumnSplit(num_columns=3, overlap_pct=0.03, header_crop_pct=0.0, footer_crop_pct=0.0),
        VlmOcr(model="Qwen/Qwen3-VL-8B-Instruct", prompt=prompts.VLM_OCR_STRUCTURED_V2, backend="transformers", max_tokens=8192, max_model_len=32768),
        MergePages(),
        ParseIssue(),
    ],
)
