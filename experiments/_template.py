from __future__ import annotations

import json
import pathlib as pl
import sys
import time
import typing as tp

sys.path.insert(0, "src")

EXP_NAME = pl.Path(__file__).stem
GROUND_TRUTH_DIR = pl.Path("eval/ground_truth")
PREDICTIONS_DIR = pl.Path("eval/predictions")


def load_pages(date: str) -> list[bytes]:
    return [p.read_bytes() for p in sorted(GROUND_TRUTH_DIR.joinpath(date).glob("*.jpeg"), key=lambda p: int(p.stem))]


def transcribe_issue(pages: list[bytes], date: str) -> list[dict[str, tp.Any]]:
    raise NotImplementedError("the experiment: pages in, article dicts out — any implementation")


def build_issue(date: str, page_count: int, articles: list[dict[str, tp.Any]]) -> dict[str, tp.Any]:
    for idx, art in enumerate(articles):
        art["id"] = f"{date}_a{idx:02d}"
        art["position_in_issue"] = idx
        art.setdefault("unit_type", "article")
        art.setdefault("headline", None)
        for p_idx, para in enumerate(art.get("paragraphs", [])):
            para["id"] = f"{date}_a{idx:02d}_p{p_idx:02d}"
    return {"date": date, "source": "il_messaggero", "page_count": page_count, "articles": articles}


def run_date(date: str) -> None:
    pages = load_pages(date)
    if not pages:
        raise SystemExit(f"no images at {GROUND_TRUTH_DIR / date}")
    t0 = time.time()
    articles = transcribe_issue(pages, date)
    elapsed = time.time() - t0
    issue = build_issue(date, len(pages), articles)
    out = PREDICTIONS_DIR / f"{EXP_NAME}_{date}.json"
    out.write_text(json.dumps(issue, indent=2, ensure_ascii=False))
    print(f"{date}: {len(articles)} articles | {elapsed:.1f}s | {elapsed / len(pages):.2f} s/page -> {out}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {EXP_NAME}.py <date> [<date> ...]")
    for date in sys.argv[1:]:
        run_date(date)


if __name__ == "__main__":
    main()
