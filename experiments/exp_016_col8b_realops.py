from __future__ import annotations

import dataclasses as dc
import json
import pathlib as pl
import sys

sys.path.insert(0, "src")

from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators.column_split import ColumnSplit
from mausoleo.ocr.operators.vlm_ocr import VlmOcr
from mausoleo.ocr.operators.merge import MergePages
from mausoleo.ocr.operators.parse import ParseIssue
from mausoleo.ocr import prompts
from mausoleo.ocr.pipeline import run_pipeline

GT_DIR = pl.Path("eval/ground_truth")
PRED_DIR = pl.Path("eval/predictions")
QWEN3 = "Qwen/Qwen3-VL-8B-Instruct"
DATES = ["1885-06-15", "1895-06-15", "1910-06-15", "1925-06-15", "1935-06-15", "1952-06-15"]


def _config() -> OcrPipelineConfig:
    return OcrPipelineConfig(
        name="exp_016_col8b_realops",
        operators=[
            ColumnSplit(num_columns=3, overlap_pct=0.03),
            VlmOcr(model=QWEN3, prompt=prompts.VLM_OCR_STRUCTURED_V2, backend="vllm",
                   max_tokens=8192, max_model_len=12288, gpu_fraction=1.0,
                   gpu_memory_utilization=0.92, vllm_strict=False),
            MergePages(),
            ParseIssue(),
        ],
    )


def main() -> None:
    config = _config()
    for date in (sys.argv[1:] or DATES):
        images = [f.read_bytes() for f in sorted(GT_DIR.joinpath(date).glob("*.jpeg"), key=lambda p: int(p.stem))]
        issue = run_pipeline(config, images, date=date)
        out = PRED_DIR / f"exp_016_col8b_realops_{date}.json"
        d = dc.asdict(issue) if dc.is_dataclass(issue) else (json.loads(issue) if isinstance(issue, str) else issue)
        out.write_text(json.dumps(d, ensure_ascii=False))
        n = len(d.get("articles", [])) if isinstance(d, dict) else "?"
        print(f"{date}: {n} articles", flush=True)


if __name__ == "__main__":
    main()
