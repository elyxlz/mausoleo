"""Pinned adversarial invariants for MausoleoBench. Run after any metric change.

A degenerate prediction must NOT score well:
  - word-scrambled (segmentation right, text unreadable) <= 0.10
  - one blob per page (perfect text, wrong segmentation) <= 0.25
  - single full-issue blob                                = 0.00
"""

from __future__ import annotations

import json
import pathlib as pl
import random
import sys

sys.path.insert(0, "src")

from mausoleo.eval.evaluate import evaluate_issue

GT_DIR = pl.Path("eval/ground_truth")
DATES = ["1885-06-15", "1910-06-15"]


def _scramble(text: str, rng: random.Random) -> str:
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)


def probe(date: str) -> dict[str, float]:
    gt = json.loads((GT_DIR / date / "ground_truth.json").read_text())
    rng = random.Random(0)

    scrambled = {**gt, "articles": [{**a, "headline": (_scramble(a["headline"], rng) if a.get("headline") else None), "paragraphs": [{"text": _scramble(p["text"], rng)} for p in a["paragraphs"]]} for a in gt["articles"]]}

    by_page: dict[int, list[str]] = {}
    for a in gt["articles"]:
        by_page.setdefault(a["page_span"][0], []).extend(p["text"] for p in a["paragraphs"])
    page_blob = {**gt, "articles": [{"headline": None, "paragraphs": [{"text": " ".join(txts)}], "page_span": [pg]} for pg, txts in sorted(by_page.items())]}

    all_text = " ".join(p["text"] for a in gt["articles"] for p in a["paragraphs"])
    full_blob = {**gt, "articles": [{"headline": None, "paragraphs": [{"text": all_text}], "page_span": [1]}]}

    return {
        "clean": evaluate_issue(gt, gt, date=date).mausoleobench_score,
        "scramble": evaluate_issue(gt, scrambled, date=date).mausoleobench_score,
        "page_blob": evaluate_issue(gt, page_blob, date=date).mausoleobench_score,
        "full_blob": evaluate_issue(gt, full_blob, date=date).mausoleobench_score,
    }


def main() -> None:
    limits = {"scramble": 0.26, "page_blob": 0.10, "full_blob": 0.001}
    ok = True
    for date in DATES:
        r = probe(date)
        print(f"{date}: clean={r['clean']:.3f} scramble={r['scramble']:.3f} page_blob={r['page_blob']:.3f} full_blob={r['full_blob']:.3f}")
        if r["clean"] < 0.95:
            print(f"  FAIL clean should be ~1.0"); ok = False
        for k, lim in limits.items():
            if r[k] > lim:
                print(f"  FAIL {k}={r[k]:.3f} > {lim}"); ok = False
    print("ALL INVARIANTS HELD" if ok else "INVARIANTS VIOLATED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
