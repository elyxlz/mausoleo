from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr

LEGACY_ENV = {"pip": ["transformers==4.44.0", "accelerate>=0.25.0", "soundfile", "librosa", "bitsandbytes"]}

config = OcrPipelineConfig(
    name="minicpm_o_4bit_v2_structured",
    operators=[
        VlmOcr(model="openbmb/MiniCPM-o-2_6", prompt=prompts.VLM_OCR_STRUCTURED_V2, backend="transformers", max_tokens=8192, load_in_4bit=True, runtime_env=LEGACY_ENV),
        MergePages(),
        ParseIssue(),
    ],
)
