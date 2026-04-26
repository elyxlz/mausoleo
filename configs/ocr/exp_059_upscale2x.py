from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergePages, ParseIssue, Preprocess, VlmOcr

config = OcrPipelineConfig(
    name="exp_059_upscale2x",
    operators=[
        Preprocess(grayscale=False, upscale=2.0, max_dimension=8000),
        ColumnSplit(num_columns=3, overlap_pct=0.03),
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
