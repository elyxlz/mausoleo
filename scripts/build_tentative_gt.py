from __future__ import annotations

import json
import pathlib as pl
import shutil
import sys
import typing as tp

PRED_DIR = pl.Path("eval/predictions")
IMAGES_DIR = pl.Path("eval/ground_truth")
TENTATIVE_DIR = pl.Path("eval/tentative_gt")

BUNDLE_SOURCES = [
    "ensemble_30min",
    "ensemble_prune5",
    "exp_157_paddleocr_titles_squeeze",
    "exp_107_fullpage_qwen25vl",
    "exp_045_qwen3vl_vllm",
    "exp_140_yolo_smallregion_vllm",
]


def load_source(name: str, date: str) -> list[dict[str, tp.Any]]:
    path = PRED_DIR / f"{name}_{date}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("articles", [])


def page_articles(articles: list[dict[str, tp.Any]], page: int) -> list[dict[str, tp.Any]]:
    return [a for a in articles if page in a.get("page_span", [])]


def format_article(article: dict[str, tp.Any]) -> str:
    headline = article.get("headline") or "(no headline)"
    unit_type = article.get("unit_type", "article")
    body = "\n".join(p.get("text", "") for p in article.get("paragraphs", []))
    return f"[{unit_type}] {headline}\n{body}"


def cmd_bundle(date: str, page: int) -> None:
    out_dir = TENTATIVE_DIR / date / "bundles"
    out_dir.mkdir(parents=True, exist_ok=True)
    sections: list[str] = [
        f"# Source transcriptions for issue {date}, page {page}",
        f"Page image: eval/ground_truth/{date}/{page}.jpeg",
    ]
    for name in BUNDLE_SOURCES:
        arts = page_articles(load_source(name, date), page)
        sections.append(f"\n{'=' * 70}\n## SOURCE: {name} ({len(arts)} articles on this page)\n{'=' * 70}")
        for i, art in enumerate(arts):
            sections.append(f"\n--- {name} article {i} ---\n{format_article(art)}")
    out_path = out_dir / f"page_{page:02d}_bundle.md"
    out_path.write_text("\n".join(sections))
    print(f"{out_path} ({out_path.stat().st_size} bytes)")


def cmd_assemble(date: str) -> None:
    issue_dir = TENTATIVE_DIR / date
    drafts = sorted(issue_dir.glob("page_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    if not drafts:
        raise SystemExit(f"no page drafts in {issue_dir}")
    articles: list[dict[str, tp.Any]] = []
    for draft in drafts:
        articles.extend(json.loads(draft.read_text()).get("articles", []))
    for idx, art in enumerate(articles):
        art["id"] = f"{date}_a{idx:02d}"
        art["position_in_issue"] = idx
        for p_idx, para in enumerate(art.get("paragraphs", [])):
            para["id"] = f"{date}_a{idx:02d}_p{p_idx:02d}"
    images = sorted(IMAGES_DIR.joinpath(date).glob("*.jpeg"), key=lambda p: int(p.stem))
    pages_dir = issue_dir / "pages"
    pages_dir.mkdir(exist_ok=True)
    for img in images:
        shutil.copy(img, pages_dir / img.name)
    issue = {"date": date, "source": "il_messaggero", "page_count": len(images), "articles": articles}
    out_path = issue_dir / "ground_truth.json"
    out_path.write_text(json.dumps(issue, indent=2, ensure_ascii=False))
    print(f"{out_path}: {len(articles)} articles, {len(images)} page images copied")


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in ("bundle", "assemble"):
        print("usage: build_tentative_gt.py bundle <date> <page> | assemble <date>")
        raise SystemExit(1)
    if sys.argv[1] == "bundle":
        cmd_bundle(sys.argv[2], int(sys.argv[3]))
    else:
        cmd_assemble(sys.argv[2])


if __name__ == "__main__":
    main()
