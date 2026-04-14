from __future__ import annotations

import json
import pathlib as pl

GT_PATH = pl.Path("eval/bootstrap_gt/1885-06-15/ground_truth.json")
gt = json.loads(GT_PATH.read_text())

total_chars = 0
for i, art in enumerate(gt["articles"]):
    text = art["paragraphs"][0]["text"] if art["paragraphs"] else ""
    total_chars += len(text)
    t = art.get("unit_type", "article")
    pages = art.get("page_span", [])
    headline = art["headline"]
    print(f"{i:2d}. [{t:13s}] p{pages} {headline[:50]:50s} ({len(text):5d} chars)")

print(f"\nTotal: {len(gt['articles'])} articles, {total_chars} chars")
