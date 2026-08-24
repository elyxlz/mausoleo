from __future__ import annotations

import base64
import concurrent.futures as cf
import dataclasses as dc
import json
import os
import pathlib as pl
import random
import re
import sys
import time
import typing as tp

sys.path.insert(0, "src")

from mausoleo.ocr.models import Issue, issue_from_dict
from mausoleo.ocr.operators.column_split import ColumnSplit, column_split
from mausoleo.ocr.operators.merge import MergePages, merge_pages
from mausoleo.ocr.operators.parse import ParseIssue, parse_issue

GT_DIR = pl.Path("eval/ground_truth")
PRED_DIR = pl.Path("eval/predictions")
DATES = ["1885-06-15", "1895-06-15", "1910-06-15", "1925-06-15", "1935-06-15", "1952-06-15"]

MODEL = "gemini-3.6-flash"
THINKING_BUDGET = 128
MAX_OUTPUT_TOKENS = 60000
CONCURRENCY = 8
MAX_TRIES = 6

PROMPT = (
    "Transcribe this column crop of a historical Italian newspaper (Il Messaggero) "
    "completely and verbatim. Transcribe every line of text; never summarize, skip or "
    "paraphrase. Preserve the original Italian exactly as printed, including archaic "
    "spelling. Split the column into its distinct content units (articles, advertisements, "
    "obituaries, notices), reading top to bottom.\n"
    'Return raw JSON: {"articles": [{"headline": string or null, "text": string}]}'
)

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_LINE_BREAK = re.compile(r"[ \t]*\n[ \t]*")


def unwrap_lines(text: str) -> str:
    return _LINE_BREAK.sub(" ", _HYPHEN_BREAK.sub(r"\1\2", text)).strip()


def unwrap_crop_json(raw: str) -> str:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return unwrap_lines(raw)
    for article in data.get("articles", []) if isinstance(data, dict) else []:
        if isinstance(article, dict) and isinstance(article.get("text"), str):
            article["text"] = unwrap_lines(article["text"])
    return json.dumps(data, ensure_ascii=False)


_COLUMNS = ColumnSplit(num_columns=3, overlap_pct=0.03)
_MERGE = MergePages()
_PARSE = ParseIssue()


def _client() -> tp.Any:
    import google.genai as genai

    key = os.environ.get("GEMINI_API_KEY") or pl.Path(os.path.expanduser("~/.gemini_key")).read_text().strip()
    return genai.Client(api_key=key)


def _is_transient(error: Exception) -> bool:
    text = str(error)
    return any(token in text for token in ("503", "429", "500", "UNAVAILABLE", "overloaded"))


def _transcribe_crop(client: tp.Any, image: bytes) -> str:
    from google.genai import types

    config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
    )
    for attempt in range(MAX_TRIES):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[types.Part.from_bytes(data=image, mime_type="image/jpeg"), PROMPT],
                config=config,
            )
            return response.text or ""
        except Exception as error:
            if not _is_transient(error) or attempt == MAX_TRIES - 1:
                raise
            time.sleep(min(2**attempt + random.random() * 3, 45))
    return ""


def _load_pages(date: str) -> list[bytes]:
    files = sorted(GT_DIR.joinpath(date).glob("*.jpeg"), key=lambda p: int(p.stem))
    return [f.read_bytes() for f in files]


def _run_issue(client: tp.Any, date: str) -> Issue:
    pages = _load_pages(date)
    row: dict[str, tp.Any] = {
        "issue_id": date,
        "date": date,
        "source": "il_messaggero",
        "page_count": len(pages),
        "images_b64": "|".join(base64.b64encode(page).decode() for page in pages),
    }
    row = column_split(row, config=_COLUMNS)
    crops = [base64.b64decode(b64) for b64 in str(row["images_b64"]).split("|")]
    with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        page_texts = [unwrap_crop_json(t) for t in pool.map(lambda crop: _transcribe_crop(client, crop), crops)]
    row = {**row, "page_texts": json.dumps(page_texts)}
    row = merge_pages(row, config=_MERGE)
    row = parse_issue(row, config=_PARSE)
    return issue_from_dict(json.loads(row["issue_json"]))


def main() -> None:
    client = _client()
    for date in sys.argv[1:] or DATES:
        start = time.time()
        issue = _run_issue(client, date)
        out = PRED_DIR / f"exp_018_gemini_flash_{date}.json"
        out.write_text(json.dumps(dc.asdict(issue), ensure_ascii=False))
        print(f"{date}: {len(issue.articles)} articles in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
