from __future__ import annotations

import base64
import dataclasses as dc
import json
import pathlib as pl
import sys
import typing as tp

sys.path.insert(0, "src")

from mausoleo.ocr import prompts
from mausoleo.ocr.models import Issue, issue_from_dict
from mausoleo.ocr.operators.column_split import ColumnSplit, column_split
from mausoleo.ocr.operators.merge import MergePages, merge_pages
from mausoleo.ocr.operators.parse import ParseIssue, parse_issue
from mausoleo.ocr.operators.vlm_ocr import VlmOcr, VlmOcrOperator

GT_DIR = pl.Path("eval/ground_truth")
PRED_DIR = pl.Path("eval/predictions")
QWEN3 = "Qwen/Qwen3-VL-8B-Instruct"
DATES = ["1885-06-15", "1895-06-15", "1910-06-15", "1925-06-15", "1935-06-15", "1952-06-15"]

_COLUMNS = ColumnSplit(num_columns=3, overlap_pct=0.03)
_VLM = VlmOcr(model=QWEN3, prompt=prompts.VLM_OCR_STRUCTURED_V2, backend="vllm",
              max_tokens=8192, max_model_len=12288, gpu_fraction=1.0,
              gpu_memory_utilization=0.92, vllm_strict=False)
_MERGE = MergePages()
_PARSE = ParseIssue()


def _encode(images: list[bytes]) -> str:
    return "|".join(base64.b64encode(img).decode() for img in images)


def _load_pages(date: str) -> list[bytes]:
    files = sorted(GT_DIR.joinpath(date).glob("*.jpeg"), key=lambda p: int(p.stem))
    return [f.read_bytes() for f in files]


def _ocr_row(vlm: VlmOcrOperator, row: dict[str, tp.Any]) -> dict[str, tp.Any]:
    batch = {key: [value] for key, value in row.items()}
    out = vlm(batch)
    return {key: value[0] for key, value in out.items()}


def _run_issue(vlm: VlmOcrOperator, date: str) -> Issue:
    row: dict[str, tp.Any] = {
        "issue_id": date,
        "date": date,
        "source": "il_messaggero",
        "page_count": len(_load_pages(date)),
        "images_b64": _encode(_load_pages(date)),
    }
    row = column_split(row, config=_COLUMNS)
    row = _ocr_row(vlm, row)
    row = merge_pages(row, config=_MERGE)
    row = parse_issue(row, config=_PARSE)
    return issue_from_dict(json.loads(row["issue_json"]))


def main() -> None:
    vlm = VlmOcrOperator(_VLM)
    for date in (sys.argv[1:] or DATES):
        issue = _run_issue(vlm, date)
        out = PRED_DIR / f"exp_017_column8b_direct_{date}.json"
        out.write_text(json.dumps(dc.asdict(issue), ensure_ascii=False))
        print(f"{date}: {len(issue.articles)} articles", flush=True)


if __name__ == "__main__":
    main()
