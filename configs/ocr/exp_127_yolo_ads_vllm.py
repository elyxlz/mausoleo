from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr, YoloCrop

config = OcrPipelineConfig(
    name="exp_127_yolo_ads_vllm",
    operators=[
        YoloCrop(conf_threshold=0.2, gpu_fraction=0.3, min_region_area=3000),
        VlmOcr(
            model="Qwen/Qwen3-VL-8B-Instruct",
            prompt=prompts.VLM_OCR_ADS_FOCUSED,
            backend="vllm",
            max_tokens=2048,
            max_model_len=13312,
            gpu_fraction=1.0,
        ),
        MergePages(),
        ParseIssue(),
    ],
)
