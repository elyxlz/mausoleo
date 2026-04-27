from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="exp_124_gemma3_4b_fullpage_vllm",
    operators=[
        VlmOcr(
            model="google/gemma-3-4b-it",
            prompt=prompts.VLM_OCR_STRUCTURED_V2,
            backend="vllm",
            max_tokens=4096,
            max_model_len=20480,
            gpu_fraction=1.0,
        ),
        MergePages(),
        ParseIssue(),
    ],
)
