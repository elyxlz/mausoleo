from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, Preprocess, VlmOcr

LEGACY_ENV = {"pip": ["transformers==4.44.0", "accelerate>=0.25.0", "bitsandbytes", "flash_attn"]}

config = OcrPipelineConfig(
    name="minicpm_o_4bit_v2_structured",
    operators=[
        Preprocess(grayscale=False, max_dimension=1024),
        VlmOcr(model="openbmb/MiniCPM-o-2_6", prompt=prompts.VLM_OCR_STRUCTURED_V2, backend="transformers", max_tokens=8192, load_in_4bit=True, runtime_env=LEGACY_ENV),
        MergePages(),
        ParseIssue(),
    ],
)
