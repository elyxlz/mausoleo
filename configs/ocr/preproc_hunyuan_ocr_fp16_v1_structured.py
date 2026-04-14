from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, Preprocess, VlmOcr

HUNYUAN_ENV = {
    "pip": [
        "transformers==4.49.0",
        "accelerate>=0.25.0",
        "flash_attn",
        "torch==2.5.1",
    ]
}

config = OcrPipelineConfig(
    name="preproc_hunyuan_ocr_fp16_v1_structured",
    operators=[
        Preprocess(grayscale=False, max_dimension=1024),
        VlmOcr(model="tencent/HunyuanOCR", prompt=prompts.VLM_OCR_STRUCTURED, backend="transformers", max_tokens=8192, runtime_env=HUNYUAN_ENV),
        MergePages(),
        ParseIssue(),
    ],
)
