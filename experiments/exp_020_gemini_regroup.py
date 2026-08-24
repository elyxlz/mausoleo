from __future__ import annotations

import concurrent.futures as cf
import dataclasses as dc
import io
import json
import os
import pathlib as pl
import random
import re
import subprocess
import sys
import time
import typing as tp

from PIL import Image

from mausoleo.ocr.models import Issue, issue_from_dict

GT_DIR = pl.Path("eval/ground_truth")
PRED_DIR = pl.Path("eval/predictions")
DATES = ["1885-06-15", "1895-06-15", "1910-06-15", "1925-06-15", "1935-06-15", "1952-06-15"]

MODEL = "gemini-3.7-flash"
THINKING_BUDGET = 128
CONCURRENCY = 10
MAX_TRIES = 6
SPAN_FACTOR = 1.7
MIN_COLUMN_FRAC = 0.03

LAYOUT_SCRIPT = r"""
import json, sys
from paddleocr import LayoutDetection
model = LayoutDetection(model_name="PP-DocLayoutV3")
out = []
for path in json.loads(sys.stdin.read()):
    boxes = []
    for res in model.predict(path, batch_size=1):
        for b in res.json["res"]["boxes"]:
            boxes.append({"label": b["label"], "score": float(b["score"]),
                          "coordinate": [float(c) for c in b["coordinate"]]})
    out.append(boxes)
print(json.dumps(out))
"""

TRANSCRIBE_PROMPT = (
    "Transcribe this column of a historical Italian newspaper verbatim, top to bottom, every line.\n"
    "Preserve the printed PARAGRAPH structure: each new paragraph begins with an indented first line. "
    "Separate paragraphs with a blank line. Join lines that are merely wrapped, and join words split "
    "by an end-of-line hyphen.\n"
    "Separate distinct content units with a line containing only ---.\n"
    "Plain text only."
)

REGROUP_PROMPT = (
    "Below are numbered text units transcribed from one page of a historical Italian newspaper, "
    "in reading order (each column top-to-bottom, then left to right).\n\n"
    "Group them into ARTICLES. An article is the most sensible SEMANTIC grouping:\n"
    "- A news story continued across a column break is ONE article: group those units together.\n"
    "- A classified RUBRIC is ONE article: 'AUTOMOBILI', 'VILLINI CASE TERRENI', 'MATRIMONIALI' each "
    "group ALL their individual adverts into a single article. Never make one article per advert.\n"
    "- Distinct rubrics stay separate articles. A standalone notice or advert outside any rubric is "
    "its own article.\n\n"
    "Give each article a headline: the printed headline if there is one, otherwise null.\n"
    "Use EVERY unit number exactly once. Do not rewrite, summarize or omit any text.\n"
    'Return a JSON array: [{"headline": string or null, "units": [int, ...]}]'
)


def _client() -> tp.Any:
    import google.genai as genai

    key = os.environ.get("GEMINI_API_KEY") or pl.Path(os.path.expanduser("~/.gemini_key")).read_text().strip()
    return genai.Client(api_key=key)


def _is_transient(error: Exception) -> bool:
    return any(token in str(error) for token in ("503", "429", "500", "UNAVAILABLE", "overloaded"))


def _call(client: tp.Any, parts: list[tp.Any], json_mode: bool) -> str:
    from google.genai import types

    config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=60000,
        thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
        **({"response_mime_type": "application/json"} if json_mode else {}),
    )
    for attempt in range(MAX_TRIES):
        try:
            return client.models.generate_content(model=MODEL, contents=parts, config=config).text or ""
        except Exception as error:
            if not _is_transient(error) or attempt == MAX_TRIES - 1:
                raise
            time.sleep(min(2**attempt + random.random() * 3, 45))
    return ""


def unwrap(text: str) -> str:
    joined = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    blocks = re.split(r"\n[ \t]*\n+", joined)
    return "\n\n".join(re.sub(r"[ \t]*\n[ \t]*", " ", b).strip() for b in blocks if b.strip()).strip()


def column_bands(boxes: list[dict[str, tp.Any]], width: int) -> list[tuple[int, int]]:
    body = [b for b in boxes if b["label"] not in ("header", "footer") and b["score"] > 0.3]
    if not body:
        return [(0, width)]
    widths = sorted(b["coordinate"][2] - b["coordinate"][0] for b in body)
    median = widths[len(widths) // 2] or width
    narrow = [b for b in body if (b["coordinate"][2] - b["coordinate"][0]) <= median * SPAN_FACTOR] or body
    merged: list[list[float]] = []
    for x1, x2 in sorted((b["coordinate"][0], b["coordinate"][2]) for b in narrow):
        if merged and x1 < merged[-1][1] - median * 0.25:
            merged[-1][1] = max(merged[-1][1], x2)
        else:
            merged.append([x1, x2])
    bands = [(int(a), int(b)) for a, b in merged if b - a >= width * MIN_COLUMN_FRAC]
    return bands or [(0, width)]


def detect_layout(page_paths: list[str]) -> list[list[dict[str, tp.Any]]]:
    proc = subprocess.run([sys.executable, "-c", LAYOUT_SCRIPT], input=json.dumps(page_paths), capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(f"layout failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def page_units(client: tp.Any, path: pl.Path, boxes: list[dict[str, tp.Any]]) -> list[str]:
    from google.genai import types

    image = Image.open(path)
    width, height = image.size
    crops = []
    for x1, x2 in column_bands(boxes, width):
        pad = int((x2 - x1) * 0.01)
        crop = image.crop((max(0, x1 - pad), int(height * 0.02), min(width, x2 + pad), int(height * 0.99)))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=95)
        crops.append(buf.getvalue())

    def transcribe(data: bytes) -> list[str]:
        raw = _call(client, [types.Part.from_bytes(data=data, mime_type="image/jpeg"), TRANSCRIBE_PROMPT], False)
        return [unwrap(u) for u in raw.split("---") if u.strip()]

    with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        per_column = list(pool.map(transcribe, crops))
    return [u for column in per_column for u in column if u]


def regroup(client: tp.Any, units: list[str]) -> list[dict[str, tp.Any]]:
    listing = "\n\n".join(f"[{i}] {u[:1500]}" for i, u in enumerate(units))
    raw = _call(client, [f"{REGROUP_PROMPT}\n\nUNITS:\n{listing}"], True)
    try:
        groups = json.loads(raw)
    except json.JSONDecodeError:
        return [{"headline": None, "units": [i]} for i in range(len(units))]
    if isinstance(groups, dict):
        groups = groups.get("articles") or groups.get("groups") or []
    cleaned, used = [], set()
    for g in groups if isinstance(groups, list) else []:
        if not isinstance(g, dict):
            continue
        idx = [i for i in g.get("units", []) if isinstance(i, int) and 0 <= i < len(units) and i not in used]
        if not idx:
            continue
        used.update(idx)
        cleaned.append({"headline": g.get("headline"), "units": sorted(idx)})
    for i in range(len(units)):
        if i not in used:
            cleaned.append({"headline": None, "units": [i]})
    return sorted(cleaned, key=lambda g: g["units"][0])


def build_issue(date: str, pages: list[tuple[int, list[str], list[dict[str, tp.Any]]]]) -> Issue:
    articles = []
    for page, units, groups in pages:
        for g in groups:
            paragraphs = [p for i in g["units"] for p in units[i].split("\n\n") if p.strip()]
            if not paragraphs:
                continue
            index = len(articles)
            articles.append(
                {
                    "id": f"{date}_a{index:03d}",
                    "headline": g["headline"],
                    "paragraphs": [{"id": f"{date}_a{index:03d}_p{j:02d}", "text": p.strip()} for j, p in enumerate(paragraphs)],
                    "page_span": [page],
                    "position_in_issue": index,
                }
            )
    return issue_from_dict({"date": date, "source": "il_messaggero", "page_count": len({p for p, _, _ in pages}), "articles": articles})


def run_issue(client: tp.Any, date: str) -> Issue:
    paths = sorted(GT_DIR.joinpath(date).glob("*.jpeg"), key=lambda p: int(p.stem))
    layouts = detect_layout([str(p) for p in paths])
    pages = []
    for path, boxes in zip(paths, layouts):
        units = page_units(client, path, boxes)
        pages.append((int(path.stem), units, regroup(client, units)))
    return build_issue(date, pages)


def main() -> None:
    client = _client()
    for date in sys.argv[1:] or DATES:
        start = time.time()
        issue = run_issue(client, date)
        out = PRED_DIR / f"exp_020_gemini_regroup_{date}.json"
        out.write_text(json.dumps(dc.asdict(issue), ensure_ascii=False))
        print(f"{date}: {len(issue.articles)} articles in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
