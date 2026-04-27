from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr, YoloCrop

config = OcrPipelineConfig(
    name="yolo_qwen25_7b_v2_structured",
    operators=[
        YoloCrop(conf_threshold=0.2, gpu_fraction=0.3, merge_vertical_gap=80, merge_horizontal_overlap=0.7),
        VlmOcr(model="Qwen/Qwen2.5-VL-7B-Instruct", prompt=prompts.VLM_OCR_STRUCTURED_V2, backend="vllm", max_tokens=8192, max_model_len=16384),
        MergePages(),
        ParseIssue(),
    ],
)
