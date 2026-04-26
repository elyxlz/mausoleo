from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="exp_116_fullpage_minicpm",
    operators=[
        VlmOcr(
            model="openbmb/MiniCPM-o-2_6",
            prompt=prompts.VLM_OCR_STRUCTURED_V2,
            backend="transformers",
            max_tokens=8192,
            max_model_len=20480,
            gpu_fraction=1.0,
        ),
        MergePages(),
        ParseIssue(),
    ],
)
