from __future__ import annotations

import importlib.util
import json
import pathlib as pl
import sys
import time
import typing as tp

sys.path.insert(0, "src")

EXP_NAME = pl.Path(__file__).stem
GROUND_TRUTH_DIR = pl.Path("eval/ground_truth")
PREDICTIONS_DIR = pl.Path("eval/predictions")
WORK_DIR = pl.Path("eval/autoresearch/semgroup")

_spec = importlib.util.spec_from_file_location("exp160", pl.Path(__file__).parent / "exp_160_ppdoclayout_headblocks.py")
assert _spec is not None and _spec.loader is not None
_exp160 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_exp160)


def cmd_dump(date: str) -> None:
    pages = sorted(GROUND_TRUTH_DIR.joinpath(date).glob("*.jpeg"), key=lambda p: int(p.stem))
    t0 = time.time()
    layout = _exp160.detect_layout([str(p) for p in pages])
    regions_per_page = [_exp160.build_regions(boxes, i + 1) for i, boxes in enumerate(layout)]
    crops, flat_regions = _exp160.crop_regions(date, regions_per_page)
    texts = _exp160.ocr_crops(crops)
    dump = []
    for idx, (text, region) in enumerate(zip(texts, flat_regions)):
        dump.append({"idx": idx, "page": region["page"], "class": region["class"], "bbox": region["bbox"], "text": text.strip()})
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    out = WORK_DIR / f"regions_{date}.json"
    out.write_text(json.dumps(dump, indent=1, ensure_ascii=False))
    print(f"{date}: {len(dump)} regions dumped in {time.time() - t0:.1f}s -> {out}")


def cmd_assemble(date: str) -> None:
    from mausoleo.ocr.operators.merge import squeeze_char_runs

    regions = {r["idx"]: r for r in json.loads((WORK_DIR / f"regions_{date}.json").read_text())}
    groups: list[dict[str, tp.Any]] = json.loads((WORK_DIR / f"groups_{date}.json").read_text())
    pages = sorted(GROUND_TRUTH_DIR.joinpath(date).glob("*.jpeg"), key=lambda p: int(p.stem))

    articles: list[dict[str, tp.Any]] = []
    used: set[int] = set()
    for group in groups:
        idxs = [i for i in group["regions"] if i in regions and i not in used]
        if not idxs:
            continue
        used.update(idxs)
        headline_idx = group.get("headline_region")
        headline_parts = [
            " ".join(regions[i]["text"].split())
            for i in idxs
            if regions[i]["class"] == "title" and (headline_idx is None or i == headline_idx or regions[i]["class"] == "title")
        ]
        headline = "\n".join(p for p in headline_parts if p)[:300] or None
        body_idxs = [i for i in idxs if regions[i]["class"] != "title"]
        paragraphs = [{"text": squeeze_char_runs(regions[i]["text"]).strip()} for i in body_idxs]
        paragraphs = [p for p in paragraphs if p["text"]]
        if not paragraphs and not headline:
            continue
        if not paragraphs:
            paragraphs = [{"text": headline}]
        page_span = sorted({regions[i]["page"] for i in idxs})
        articles.append(
            {
                "unit_type": group.get("unit_type", "article"),
                "headline": headline,
                "paragraphs": paragraphs,
                "page_span": page_span,
            }
        )

    for idx, art in enumerate(articles):
        art["id"] = f"{date}_a{idx:02d}"
        art["position_in_issue"] = idx
        for p_idx, para in enumerate(art["paragraphs"]):
            para["id"] = f"{date}_a{idx:02d}_p{p_idx:02d}"
    issue = {"date": date, "source": "il_messaggero", "page_count": len(pages), "articles": articles}
    out = PREDICTIONS_DIR / f"{EXP_NAME}_{date}.json"
    out.write_text(json.dumps(issue, indent=2, ensure_ascii=False))
    orphans = len(regions) - len(used)
    print(f"{date}: {len(articles)} articles from {len(groups)} groups ({orphans} orphan regions) -> {out}")


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in ("dump", "assemble"):
        raise SystemExit(f"usage: {EXP_NAME}.py dump <date> | assemble <date>")
    if sys.argv[1] == "dump":
        for date in sys.argv[2:]:
            cmd_dump(date)
    else:
        for date in sys.argv[2:]:
            cmd_assemble(date)


if __name__ == "__main__":
    main()
