from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import MergeMarkdownPages, ParseIssue
from mausoleo.ocr.operators.unlimited_ocr import UnlimitedOcr

config = OcrPipelineConfig(
    name="exp_146_unlimited_multipage",
    operators=[
        UnlimitedOcr(),
        MergeMarkdownPages(),
        ParseIssue(),
    ],
)
