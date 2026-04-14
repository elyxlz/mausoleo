from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, Preprocess, VlmOcr

config = OcrPipelineConfig(
    name="chandra_ocr2_v2_structured",
    operators=[
        Preprocess(grayscale=False, max_dimension=768),
        VlmOcr(model="datalab-to/chandra-ocr-2", prompt=prompts.VLM_OCR_STRUCTURED_V2, backend="transformers", max_tokens=8192),
        MergePages(),
        ParseIssue(),
    ],
)
