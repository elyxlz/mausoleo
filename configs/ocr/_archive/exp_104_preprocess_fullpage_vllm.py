from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, Preprocess, VlmOcr

config = OcrPipelineConfig(
    name="exp_104_preprocess_fullpage_vllm",
    operators=[
        Preprocess(grayscale=True, max_dimension=1600),
        VlmOcr(
            model="Qwen/Qwen3-VL-8B-Instruct",
            prompt=prompts.VLM_OCR_STRUCTURED_V2,
            backend="vllm",
            max_tokens=8192,
            max_model_len=20480,
            gpu_fraction=1.0,
        ),
        MergePages(),
        ParseIssue(),
    ],
)
