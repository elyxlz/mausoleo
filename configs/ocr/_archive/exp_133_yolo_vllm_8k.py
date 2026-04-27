from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr, YoloCrop

config = OcrPipelineConfig(
    name="exp_133_yolo_vllm_8k",
    operators=[
        YoloCrop(conf_threshold=0.2, gpu_fraction=0.3, min_region_area=3000),
        VlmOcr(
            model="Qwen/Qwen3-VL-8B-Instruct",
            prompt=prompts.VLM_OCR_STRUCTURED_V2,
            backend="vllm",
            max_tokens=8192,
            max_model_len=16384,
            gpu_fraction=1.0,
        ),
        MergePages(),
        ParseIssue(),
    ],
)
