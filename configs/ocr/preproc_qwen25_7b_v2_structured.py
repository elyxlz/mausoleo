from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, Preprocess, VlmOcr

config = OcrPipelineConfig(
    name="preproc_qwen25_7b_v2_structured",
    operators=[
        Preprocess(grayscale=True, max_dimension=2000),
        VlmOcr(model="Qwen/Qwen2.5-VL-7B-Instruct", prompt=prompts.VLM_OCR_STRUCTURED_V2, backend="vllm", max_tokens=8192, max_model_len=16384),
        MergePages(),
        ParseIssue(),
    ],
)
