from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergePages, ParseIssue, VlmOcr

NANONETS_ENV = {"pip": ["transformers==4.50.0", "accelerate>=0.25.0"]}

config = OcrPipelineConfig(
    name="exp_121_nanonets_transformers",
    operators=[
        ColumnSplit(num_columns=3, overlap_pct=0.03),
        VlmOcr(
            model="nanonets/Nanonets-OCR2-3B",
            prompt=prompts.VLM_OCR_STRUCTURED_V2,
            backend="transformers",
            max_tokens=8192,
            max_model_len=12288,
            gpu_fraction=1.0,
            runtime_env=NANONETS_ENV,
        ),
        MergePages(),
        ParseIssue(),
    ],
)
