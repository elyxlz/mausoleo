from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergeMarkdownPages, ParseIssue, VlmOcr

HUNYUAN_PARSE_PROMPT = "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"

HUNYUAN_ENV = {
    "pip": [
        "transformers>=5.13.0",
        "accelerate>=0.25.0",
    ]
}

config = OcrPipelineConfig(
    name="exp_150_hunyuan_col3_native",
    operators=[
        ColumnSplit(num_columns=3, overlap_pct=0.03),
        VlmOcr(
            model="tencent/HunyuanOCR",
            prompt=HUNYUAN_PARSE_PROMPT,
            backend="transformers",
            max_tokens=8192,
            runtime_env=HUNYUAN_ENV,
        ),
        MergeMarkdownPages(),
        ParseIssue(),
    ],
)
