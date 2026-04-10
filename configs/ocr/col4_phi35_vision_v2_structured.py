from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergePages, ParseIssue, VlmOcr

LEGACY_ENV = {"pip": ["transformers==4.44.0", "accelerate>=0.25.0"]}

config = OcrPipelineConfig(
    name="col4_phi35_vision_v2_structured",
    operators=[
        ColumnSplit(num_columns=4, overlap_pct=0.03),
        VlmOcr(model="microsoft/Phi-3.5-vision-instruct", prompt=prompts.VLM_OCR_STRUCTURED_V2, backend="transformers", max_tokens=8192, runtime_env=LEGACY_ENV),
        MergePages(),
        ParseIssue(),
    ],
)
