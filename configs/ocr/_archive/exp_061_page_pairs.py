from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, PagePairVlm, ParseIssue

config = OcrPipelineConfig(
    name="exp_061_page_pairs",
    operators=[
        PagePairVlm(
            model="Qwen/Qwen3-VL-8B-Instruct",
            prompt=prompts.VLM_OCR_PAGE_PAIR,
            window_size=2,
            stride=1,
            max_tokens=8192,
            max_model_len=15360,
            gpu_fraction=1.0,
        ),
        ParseIssue(),
    ],
)
