from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ChandraLayout, MergePages, ParseIssue, VlmOcr

config = OcrPipelineConfig(
    name="exp_041_chandra_layout_qwen3",
    operators=[
        ChandraLayout(gpu_fraction=0.5),
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
