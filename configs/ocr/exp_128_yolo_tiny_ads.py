from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr, YoloCrop

config = OcrPipelineConfig(
    name="exp_128_yolo_tiny_ads",
    operators=[
        YoloCrop(conf_threshold=0.10, gpu_fraction=0.3, min_region_area=500, padding=15),
        VlmOcr(
            model="Qwen/Qwen3-VL-8B-Instruct",
            prompt=prompts.VLM_OCR_ADS_FOCUSED,
            backend="vllm",
            max_tokens=1024,
            max_model_len=14336,
            gpu_fraction=1.0,
        ),
        MergePages(),
        ParseIssue(),
    ],
)
