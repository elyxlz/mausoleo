import os

from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergePages, ParseIssue, Preprocess, VlmOcr

CHANDRA_ENV = {
    "pip": [
        "transformers>=5.5.0",
        "accelerate>=0.25.0",
        "qwen-vl-utils",
        "safetensors",
    ],
    "env_vars": {
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "HF_HOME": os.environ.get("HF_HOME", "/tmp/hf_cache"),
        "PIP_FIND_LINKS": "/tmp/pip_cache_chandra",
    },
}

config = OcrPipelineConfig(
    name="chandra_ocr2_v2_structured",
    operators=[
        Preprocess(grayscale=False, max_dimension=768),
        VlmOcr(model="datalab-to/chandra-ocr-2", prompt=prompts.VLM_OCR_STRUCTURED_V2, backend="transformers", max_tokens=8192, runtime_env=CHANDRA_ENV),
        MergePages(),
        ParseIssue(),
    ],
)
