from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr, YoloCrop

config = OcrPipelineConfig(
    name="exp_140_yolo_smallregion_vllm",
    operators=[
        YoloCrop(conf_threshold=0.15, gpu_fraction=0.3, min_region_area=1500, merge_vertical_gap=50, merge_horizontal_overlap=0.5, padding=15),
        VlmOcr(
            model="Qwen/Qwen3-VL-8B-Instruct",
            prompt=prompts.VLM_OCR_STRUCTURED_V2,
            backend="vllm",
            max_tokens=8192,
            max_model_len=16384,
            gpu_fraction=1.0,
            vllm_strict=True,
        ),
        MergePages(),
        ParseIssue(),
    ],
)
