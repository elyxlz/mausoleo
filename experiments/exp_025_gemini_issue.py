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
WEAK_PAGE_RATIO = 0.35
WEAK_PAGE_RETRIES = 2
SNIPPET_CHARS = 500

PROMPT = (
    "You are an expert OCR system for historical Italian newspapers. Read each column "
    "top-to-bottom, then left-to-right. Transcribe ALL text; do not skip or summarize. "
    "Preserve archaic spelling. Separate distinct content units.\n"
    'Return a JSON array: [{"headline": string or null, "text": string}]'
)

REGROUP_PROMPT = (
    "Below are numbered text units transcribed from a WHOLE ISSUE of a historical Italian newspaper, "
    "in reading order. Each is tagged with its page.\n\n"
    "Some units are fragments of the same article split at a column break, and some rubrics have "
    "been split into individual entries. Merge them into correct ARTICLES.\n"
    "An article is the most sensible SEMANTIC grouping:\n"
    "- A story continued across a column break is ONE article.\n"
    "- A classified RUBRIC is ONE article: 'AUTOMOBILI', 'VILLINI CASE TERRENI', 'MATRIMONIALI' each "
    "hold ALL their adverts in a single article. Never one article per advert.\n"
    "- Distinct rubrics stay separate from each other.\n"
    "- An advert or notice printed on its own, NOT under a rubric heading, is its OWN article. Only "
    "group adverts together when they sit under a shared rubric heading on the page.\n"
    "- A RUBRIC HEADING (e.g. 'AUTOMOBILI BICICL. (L. 90)', 'OCCASIONI (L. 85)', 'DOMANDE LAVORO') "
    "absorbs EVERY advert that follows it, in order, until the next rubric heading appears. Put the "
    "heading unit and all of those adverts into ONE article. These runs are long -- often dozens of "
    "units -- so keep going until the next heading.\n"
    "- A story or rubric that runs past the bottom of a page CONTINUES on the next page: group those "
    "units together even though their page numbers differ. Text breaking off mid-sentence is the "
    "signal.\n"
    "- Most units are already correct on their own — only merge when the page clearly calls for it.\n\n"
    "Give each article the printed headline, or null.\n"
    "Use EVERY unit number exactly once. Never rewrite, summarize or omit text.\n"
    'Return a JSON array: [{"headline": string or null, "units": [int, ...]}]'
)

_MERGE = MergePages()
_PARSE = ParseIssue()

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_PARA_BREAK = re.compile(r"[ \t]*\n+[ \t]*")
_LINE_BREAK = re.compile(r"[ \t]*\n[ \t]*")


def unwrap_lines(text: str) -> str:
    return _LINE_BREAK.sub(" ", _HYPHEN_BREAK.sub(r"\1\2", text)).strip()


def article_headline(article: dict[str, tp.Any]) -> str | None:
    raw = article.get("headline")
    if isinstance(raw, list):
        raw = " ".join(str(x) for x in raw if x)
    text = unwrap_lines(str(raw)) if raw else ""
    return text or None


def article_paragraphs(article: dict[str, tp.Any]) -> list[str]:
    raw = article.get("paragraphs")
    if isinstance(raw, list) and raw:
        parts = [str(p) for p in raw if str(p).strip()]
    else:
        parts = _PARA_BREAK.split(_HYPHEN_BREAK.sub(r"\1\2", str(article.get("text") or "")))
    return [cleaned for part in parts for cleaned in [unwrap_lines(part)] if cleaned]


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
    part = types.Part.from_bytes(data=image, mime_type="image/jpeg")
    part.media_resolution = types.PartMediaResolution(level=types.PartMediaResolutionLevel.MEDIA_RESOLUTION_ULTRA_HIGH)
    for attempt in range(MAX_TRIES):
        try:
            response = client.models.generate_content(model=MODEL, contents=[part, PROMPT], config=config)
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


def _regroup(client: tp.Any, articles: list[dict[str, tp.Any]]) -> list[dict[str, tp.Any]]:
    if len(articles) < 2:
        return articles
    listing = "\n\n".join(
        f"[{i}] {(a.get('headline') or '')} :: " + " ".join(p["text"] for p in a["paragraphs"])[:900] for i, a in enumerate(articles)
    )
    raw = _call_text(client, f"{REGROUP_PROMPT}\n\nUNITS:\n{listing}")
    try:
        groups = json.loads(raw)
    except json.JSONDecodeError:
        return articles
    if isinstance(groups, dict):
        groups = groups.get("articles") or groups.get("groups") or []
    merged, used = [], set()
    for group in groups if isinstance(groups, list) else []:
        if not isinstance(group, dict):
            continue
        idx = [i for i in group.get("units", []) if isinstance(i, int) and 0 <= i < len(articles) and i not in used]
        if not idx:
            continue
        used.update(idx)
        paragraphs = [p for i in sorted(idx) for p in articles[i]["paragraphs"]]
        headline = group.get("headline") or next((articles[i].get("headline") for i in sorted(idx) if articles[i].get("headline")), None)
        span = sorted({(articles[i].get("page_span") or [1])[0] for i in idx})
        merged.append({"headline": headline, "paragraphs": paragraphs, "page_span": span, "_first": min(idx)})
    for i, a in enumerate(articles):
        if i not in used:
            merged.append({**a, "_first": i})
    merged.sort(key=lambda a: a["_first"])
    return [{k: v for k, v in a.items() if k != "_first"} for a in merged]


def _call_text(client: tp.Any, prompt: str) -> str:
    from google.genai import types

    config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
    )
    for attempt in range(MAX_TRIES):
        try:
            return client.models.generate_content(model=MODEL, contents=[prompt], config=config).text or ""
        except Exception as error:
            if not _is_transient(error) or attempt == MAX_TRIES - 1:
                raise
            time.sleep(min(2**attempt + random.random() * 3, 45))
    return ""


def _page_chars(raw: str) -> int:
    return sum(len(str(a.get("text") or "")) for a in page_articles(raw) if isinstance(a, dict))


def _retry_weak_pages(client: tp.Any, pages: list[bytes], texts: list[str]) -> list[str]:
    for _ in range(WEAK_PAGE_RETRIES):
        counts = [_page_chars(t) for t in texts]
        healthy = sorted(c for c in counts if c > 0)
        if not healthy:
            return texts
        median = healthy[len(healthy) // 2]
        weak = [i for i, c in enumerate(counts) if c < median * WEAK_PAGE_RATIO]
        if not weak:
            return texts
        print(f"  retrying {len(weak)} weak page(s): " + ", ".join(f"p{i + 1}={counts[i]}c vs median {median}c" for i in weak), flush=True)
        with cf.ThreadPoolExecutor(max_workers=min(len(weak), CONCURRENCY)) as pool:
            fresh = list(pool.map(lambda i: _transcribe_page(client, pages[i]), weak))
        improved = False
        for i, text in zip(weak, fresh):
            if _page_chars(text) > counts[i]:
                texts[i] = text
                improved = True
        if not improved:
            return texts
    return texts


def _load_pages(date: str) -> list[bytes]:
    files = sorted(GT_DIR.joinpath(date).glob("*.jpeg"), key=lambda p: int(p.stem))
    return [f.read_bytes() for f in files]


def _run_issue(client: tp.Any, date: str) -> Issue:
    pages = _load_pages(date)
    with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        raw_texts = list(pool.map(lambda page: _transcribe_page(client, page), pages))
    raw_texts = _retry_weak_pages(client, pages, raw_texts)
    page_texts = [unwrap_page_json(text) for text in raw_texts]
    row: dict[str, tp.Any] = {
        "issue_id": date,
        "date": date,
        "source": "il_messaggero",
        "page_count": len(pages),
        "page_texts": json.dumps(page_texts),
    }
    row = merge_pages(row, config=_MERGE)
    row = parse_issue(row, config=_PARSE)
    issue = json.loads(row["issue_json"])
    ordered = sorted(issue["articles"], key=lambda a: ((a.get("page_span") or [1])[0], a.get("position_in_issue", 0)))
    regrouped = _regroup(client, ordered)
    rebuilt = []
    for article in regrouped:
        index = len(rebuilt)
        rebuilt.append(
            {
                "id": f"{date}_a{index:03d}",
                "headline": article.get("headline"),
                "paragraphs": [{"id": f"{date}_a{index:03d}_p{j:02d}", "text": p["text"]} for j, p in enumerate(article["paragraphs"])],
                "page_span": article.get("page_span") or [1],
                "position_in_issue": index,
            }
        )
    issue["articles"] = rebuilt
    return issue_from_dict(issue)


def main() -> None:
    client = _client()
    for date in sys.argv[1:] or DATES:
        start = time.time()
        issue = _run_issue(client, date)
        out = PRED_DIR / f"exp_025_gemini_issue_{date}.json"
        out.write_text(json.dumps(dc.asdict(issue), ensure_ascii=False))
        print(f"{date}: {len(issue.articles)} articles in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
