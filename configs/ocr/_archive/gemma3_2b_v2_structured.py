from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, Preprocess, VlmOcr

config = OcrPipelineConfig(
    name="gemma3_2b_v2_structured",
    operators=[
        Preprocess(grayscale=False, max_dimension=1024),
        VlmOcr(model="google/gemma-3-4b-it", prompt=prompts.VLM_OCR_STRUCTURED_V2, backend="transformers", max_tokens=8192, max_model_len=16384),
        MergePages(),
        ParseIssue(),
    ],
)
