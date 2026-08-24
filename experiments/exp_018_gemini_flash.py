from __future__ import annotations

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
from mausoleo.ocr.operators.merge import MergePages, merge_pages
from mausoleo.ocr.operators.parse import ParseIssue, parse_issue

GT_DIR = pl.Path("eval/ground_truth")
PRED_DIR = pl.Path("eval/predictions")
DATES = ["1885-06-15", "1895-06-15", "1910-06-15", "1925-06-15", "1935-06-15", "1952-06-15"]

MODEL = "gemini-3.7-flash"
THINKING_BUDGET = 128
MAX_OUTPUT_TOKENS = 60000
CONCURRENCY = 12
MAX_TRIES = 6

PROMPT = (
    "You are an expert OCR system for historical Italian newspapers. Read each column "
    "top-to-bottom, then left-to-right. Transcribe ALL text; do not skip or summarize. "
    "Preserve archaic spelling. Separate distinct content units.\n"
    "Preserve the printed paragraph structure: each paragraph begins with an indented first "
    "line in the original. Emit one array entry per printed paragraph. Join lines that are "
    "merely wrapped, and join words split by an end-of-line hyphen.\n"
    'Return a JSON array: [{"headline": string or null, "paragraphs": [string, ...]}]'
)

_MERGE = MergePages()
_PARSE = ParseIssue()

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_PARA_BREAK = re.compile(r"\n[ \t]*\n+")
_LINE_BREAK = re.compile(r"[ \t]*\n[ \t]*")


def unwrap_lines(text: str) -> str:
    joined = _HYPHEN_BREAK.sub(r"\1\2", text)
    return "\n\n".join(_LINE_BREAK.sub(" ", part).strip() for part in _PARA_BREAK.split(joined) if part.strip()).strip()


def article_headline(article: dict[str, tp.Any]) -> str | None:
    raw = article.get("headline")
    if isinstance(raw, list):
        raw = " ".join(str(x) for x in raw if x)
    text = unwrap_lines(str(raw)) if raw else ""
    return text or None


def article_paragraphs(article: dict[str, tp.Any]) -> list[str]:
    raw = article.get("paragraphs")
    if isinstance(raw, list) and raw:
        parts = [unwrap_lines(str(p)) for p in raw if str(p).strip()]
    else:
        parts = unwrap_lines(str(article.get("text") or "")).split("\n\n")
    return [p.strip() for p in parts if p.strip()]


def page_articles(raw: str) -> list[tp.Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        return data.get("articles", [])
    return data if isinstance(data, list) else []


def unwrap_page_json(raw: str) -> str:
    articles = page_articles(raw)
    if not articles:
        return unwrap_lines(raw)
    normalised = []
    for article in articles:
        if not isinstance(article, dict):
            continue
        paragraphs = article_paragraphs(article)
        if not paragraphs:
            continue
        normalised.append({"headline": article_headline(article), "paragraphs": [{"text": p} for p in paragraphs]})
    return json.dumps(normalised, ensure_ascii=False)


def _client() -> tp.Any:
    import google.genai as genai

    key = os.environ.get("GEMINI_API_KEY") or pl.Path(os.path.expanduser("~/.gemini_key")).read_text().strip()
    return genai.Client(api_key=key)


def _is_transient(error: Exception) -> bool:
    return any(token in str(error) for token in ("503", "429", "500", "UNAVAILABLE", "overloaded"))


def _transcribe_page(client: tp.Any, image: bytes) -> str:
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
            text = response.text or ""
            if _article_count(text) > 0:
                return text
        except Exception as error:
            if not _is_transient(error) or attempt == MAX_TRIES - 1:
                raise
        time.sleep(min(2**attempt + random.random() * 3, 45))
    return ""


def _article_count(raw: str) -> int:
    return len(page_articles(raw))


def _load_pages(date: str) -> list[bytes]:
    files = sorted(GT_DIR.joinpath(date).glob("*.jpeg"), key=lambda p: int(p.stem))
    return [f.read_bytes() for f in files]


def _run_issue(client: tp.Any, date: str) -> Issue:
    pages = _load_pages(date)
    with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        page_texts = [unwrap_page_json(text) for text in pool.map(lambda page: _transcribe_page(client, page), pages)]
    row: dict[str, tp.Any] = {
        "issue_id": date,
        "date": date,
        "source": "il_messaggero",
        "page_count": len(pages),
        "page_texts": json.dumps(page_texts),
    }
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
