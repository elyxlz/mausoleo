from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergePages, ParseIssue, VlmOcr

HUNYUAN_ENV = {
    "pip": [
        "git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4",
        "accelerate>=0.25.0",
        "flash_attn",
        "torch==2.5.1",
        "bitsandbytes",
    ]
}

config = OcrPipelineConfig(
    name="col4_hunyuan_ocr_v1_structured",
    operators=[
        ColumnSplit(num_columns=4, overlap_pct=0.03),
        VlmOcr(model="tencent/HunyuanOCR", prompt=prompts.VLM_OCR_STRUCTURED, backend="transformers", max_tokens=8192, load_in_4bit=True, runtime_env=HUNYUAN_ENV),
        MergePages(),
        ParseIssue(),
    ],
)
