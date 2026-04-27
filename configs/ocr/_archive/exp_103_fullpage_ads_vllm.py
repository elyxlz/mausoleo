from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="exp_103_fullpage_ads_vllm",
    operators=[
        VlmOcr(
            model="Qwen/Qwen3-VL-8B-Instruct",
            prompt=prompts.VLM_OCR_ADS_FOCUSED,
            backend="vllm",
            max_tokens=8192,
            max_model_len=20480,
            gpu_fraction=1.0,
        ),
        MergePages(),
        ParseIssue(),
    ],
)
