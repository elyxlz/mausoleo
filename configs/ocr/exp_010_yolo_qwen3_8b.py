from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr, YoloCrop

config = OcrPipelineConfig(
    name="exp_010_yolo_qwen3_8b",
    operators=[
        YoloCrop(conf_threshold=0.2, gpu_fraction=0.3, min_region_area=3000),
        VlmOcr(model="Qwen/Qwen3-VL-8B-Instruct", prompt=prompts.VLM_OCR_STRUCTURED_V2, backend="transformers", max_tokens=8192, max_model_len=32768),
        MergePages(),
        ParseIssue(),
    ],
)
