from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergePages, ParseIssue, VlmOcr

INTERNVL_ENV = {"pip": ["torch==2.10.0", "transformers==4.49.0", "accelerate>=0.25.0", "timm", "einops"]}

config = OcrPipelineConfig(
    name="exp_122_internvl3_runtime_env",
    operators=[
        ColumnSplit(num_columns=3, overlap_pct=0.03),
        VlmOcr(
            model="OpenGVLab/InternVL3-8B",
            prompt=prompts.VLM_OCR_STRUCTURED_V2,
            backend="transformers",
            max_tokens=8192,
            max_model_len=32768,
            gpu_fraction=1.0,
            runtime_env=INTERNVL_ENV,
        ),
        MergePages(),
        ParseIssue(),
    ],
)
