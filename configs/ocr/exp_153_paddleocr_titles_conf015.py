from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr, YoloCrop

config = OcrPipelineConfig(
    name="exp_153_paddleocr_titles_conf015",
    operators=[
        YoloCrop(conf_threshold=0.15, separate_title_regions=True),
        VlmOcr(
            model="PaddlePaddle/PaddleOCR-VL-1.6",
            prompt="OCR:",
            backend="vllm",
            max_tokens=4096,
            max_model_len=16384,
            gpu_fraction=1.0,
            vllm_strict=True,
        ),
        MergePages(title_class_headlines=True),
        ParseIssue(),
    ],
)
