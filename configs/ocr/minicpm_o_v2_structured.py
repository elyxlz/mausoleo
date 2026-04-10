from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr

LEGACY_ENV = {"pip": ["transformers==4.44.0", "accelerate>=0.25.0", "soundfile", "librosa"]}

config = OcrPipelineConfig(
    name="minicpm_o_v2_structured",
    operators=[
        VlmOcr(model="openbmb/MiniCPM-o-2_6", prompt=prompts.VLM_OCR_STRUCTURED_V2, backend="transformers", max_tokens=8192, runtime_env=LEGACY_ENV),
        MergePages(),
        ParseIssue(),
    ],
)
