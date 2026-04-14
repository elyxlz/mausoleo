from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergePages, ParseIssue, Preprocess, VlmOcr

config = OcrPipelineConfig(
    name="col3_gemma3_9b_v2_structured",
    operators=[
        Preprocess(grayscale=False, max_dimension=512),
        ColumnSplit(num_columns=3, overlap_pct=0.03),
        VlmOcr(model="google/gemma-3-12b-it", prompt=prompts.VLM_OCR_STRUCTURED_V2, backend="transformers", max_tokens=8192, max_model_len=16384, gpu_fraction=2.0),
        MergePages(),
        ParseIssue(),
    ],
)
