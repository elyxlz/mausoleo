from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="minicpm_o_v1_structured",
    operators=[
        VlmOcr(model="openbmb/MiniCPM-o-2_6", prompt=prompts.VLM_OCR_STRUCTURED, backend="transformers", max_tokens=4096),
        MergePages(),
        ParseIssue(),
    ],
)
