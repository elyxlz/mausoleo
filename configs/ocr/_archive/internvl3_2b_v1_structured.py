from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="internvl3_2b_v1_structured",
    operators=[
        VlmOcr(model="OpenGVLab/InternVL3-2B", prompt=prompts.VLM_OCR_STRUCTURED, backend="transformers", max_tokens=4096, max_model_len=16384),
        MergePages(),
        ParseIssue(),
    ],
)
