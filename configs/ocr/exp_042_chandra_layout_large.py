from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ChandraLayout, MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="exp_042_chandra_layout_large",
    operators=[
        ChandraLayout(gpu_fraction=0.5, min_region_area=8000, merge_vertical_gap=100),
        VlmOcr(
            model="Qwen/Qwen3-VL-8B-Instruct",
            prompt=prompts.VLM_OCR_STRUCTURED_V2,
            backend="transformers",
            max_tokens=8192,
            max_model_len=32768,
        ),
        MergePages(),
        ParseIssue(),
    ],
)
