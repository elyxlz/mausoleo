from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr, YoloCrop

config = OcrPipelineConfig(
    name="exp_148_paddleocr_yolo",
    operators=[
        YoloCrop(),
        VlmOcr(
            model="PaddlePaddle/PaddleOCR-VL-1.6",
            prompt="OCR:",
            backend="vllm",
            max_tokens=4096,
            max_model_len=16384,
            gpu_fraction=1.0,
            vllm_strict=True,
        ),
        MergePages(),
        ParseIssue(),
    ],
)
