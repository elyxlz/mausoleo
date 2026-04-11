from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergePages, ParseIssue, VlmOcr

LEGACY_ENV = {"pip": ["transformers==4.44.0", "accelerate>=0.25.0", "soundfile", "librosa", "flash_attn", "vector_quantize_pytorch", "vocos"]}

config = OcrPipelineConfig(
    name="col4_minicpm_o_v2_structured",
    operators=[
        ColumnSplit(num_columns=4, overlap_pct=0.03),
        VlmOcr(model="openbmb/MiniCPM-o-2_6", prompt=prompts.VLM_OCR_STRUCTURED_V2, backend="transformers", max_tokens=8192, runtime_env=LEGACY_ENV),
        MergePages(),
        ParseIssue(),
    ],
)
