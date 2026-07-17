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


def cmd_stitch(date: str) -> None:
    issue_dir = TENTATIVE_DIR / date
    gt_path = issue_dir / "ground_truth.json"
    stitches_path = issue_dir / "stitches.json"
    issue = json.loads(gt_path.read_text())
    merges: list[list[str]] = json.loads(stitches_path.read_text())
    by_id = {a["id"]: a for a in issue["articles"]}
    absorbed: set[str] = set()
    for keep_id, absorb_id in merges:
        keep, absorb = by_id[keep_id], by_id[absorb_id]
        keep["paragraphs"] = [*keep["paragraphs"], *absorb["paragraphs"]]
        keep["page_span"] = sorted(set(keep["page_span"]) | set(absorb["page_span"]))
        absorbed.add(absorb_id)
    articles = [a for a in issue["articles"] if a["id"] not in absorbed]
    for idx, art in enumerate(articles):
        art["id"] = f"{date}_a{idx:02d}"
        art["position_in_issue"] = idx
        for p_idx, para in enumerate(art.get("paragraphs", [])):
            para["id"] = f"{date}_a{idx:02d}_p{p_idx:02d}"
    issue["articles"] = articles
    gt_path.write_text(json.dumps(issue, indent=2, ensure_ascii=False))
    print(f"{gt_path}: applied {len(merges)} stitches -> {len(articles)} articles")


def _unit_header(article: dict[str, tp.Any]) -> str:
    pages = ",".join(str(n) for n in article["page_span"])
    return f"## {article['id'].split('_')[-1]} | {article['unit_type']} | pages {pages}"


def cmd_render(date: str) -> None:
    issue_dir = TENTATIVE_DIR / date
    issue = json.loads((issue_dir / "ground_truth.json").read_text())
    md_lines: list[str] = [
        f"# {date} — tentative GT review transcript",
        "",
        "Edit text freely. Unit header format: `## aNN | unit_type | pages N[,N]`.",
        "Headline lines are `> ` blockquotes (one per printed head-block line); no blockquote = no headline.",
        "Paragraphs are blocks separated by blank lines. Delete a unit by deleting its section; add one with a `## new | article | pages N` header.",
        f"Apply edits back with: scripts/build_tentative_gt.py ingest {date}",
        "",
    ]
    by_page: dict[int, list[dict[str, tp.Any]]] = {}
    for article in issue["articles"]:
        by_page.setdefault(article["page_span"][0], []).append(article)

    html_units: list[str] = []
    for page in sorted(by_page):
        md_lines.append(f"----- PAGE {page} " + "-" * 50)
        md_lines.append("")
        unit_html: list[str] = []
        for article in by_page[page]:
            md_lines.append(_unit_header(article))
            if article.get("headline"):
                for line in article["headline"].split("\n"):
                    md_lines.append(f"> {line}")
            md_lines.append("")
            for para in article["paragraphs"]:
                md_lines.append(para["text"].strip())
                md_lines.append("")
            head_html = "".join(
                f"<div class='hl'>{line}</div>" for line in (article.get("headline") or "").split("\n") if line
            )
            paras_html = "".join(f"<p>{para['text']}</p>" for para in article["paragraphs"])
            meta = f"{article['id'].split('_')[-1]} · {article['unit_type']} · pages {','.join(str(n) for n in article['page_span'])}"
            unit_html.append(f"<div class='unit'><div class='meta'>{meta}</div>{head_html}{paras_html}</div>")
        html_units.append(
            f"<section><div class='imgpane'><img src='pages/{page}.jpeg'></div><div class='textpane'>{''.join(unit_html)}</div></section>"
        )

    (issue_dir / "REVIEW.md").write_text("\n".join(md_lines))
    html = (
        "<!doctype html><meta charset='utf-8'><title>" + date + " review</title><style>"
        "body{font-family:Georgia,serif;margin:0}section{display:flex;border-bottom:4px solid #333}"
        ".imgpane{width:55%;position:sticky;top:0;align-self:flex-start;max-height:100vh;overflow:auto}"
        ".imgpane img{width:100%}.textpane{width:45%;padding:1em 2em;box-sizing:border-box}"
        ".unit{margin-bottom:1.6em;border-left:3px solid #ccc;padding-left:.8em}"
        ".meta{font-family:monospace;font-size:.75em;color:#888}.hl{font-weight:bold;font-size:1.1em}"
        "p{margin:.4em 0;line-height:1.35}</style>" + "".join(html_units)
    )
    (issue_dir / "REVIEW.html").write_text(html)
    print(f"{issue_dir}/REVIEW.md + REVIEW.html ({sum(len(v) for v in by_page.values())} units)")


def cmd_ingest(date: str) -> None:
    import re

    issue_dir = TENTATIVE_DIR / date
    issue = json.loads((issue_dir / "ground_truth.json").read_text())
    lines = (issue_dir / "REVIEW.md").read_text().split("\n")
    header_re = re.compile(r"^## (\S+) \| (\w+) \| pages? ([\d, ]+)$")

    articles: list[dict[str, tp.Any]] = []
    current: dict[str, tp.Any] | None = None
    block: list[str] = []

    def flush_block() -> None:
        if current is not None and block:
            text = "\n".join(block).strip()
            if text:
                current["paragraphs"].append({"text": text})
        block.clear()

    for line in lines:
        m = header_re.match(line)
        if m:
            flush_block()
            if current is not None:
                articles.append(current)
            pages = [int(x) for x in m.group(3).replace(" ", "").split(",") if x]
            current = {"unit_type": m.group(2), "headline": None, "paragraphs": [], "page_span": pages}
            continue
        if current is None or line.startswith("----- PAGE"):
            flush_block()
            continue
        if line.startswith("> "):
            flush_block()
            head_line = line[2:].strip()
            current["headline"] = head_line if current["headline"] is None else f"{current['headline']}\n{head_line}"
            continue
        if line.strip() == "":
            flush_block()
        else:
            block.append(line)
    flush_block()
    if current is not None:
        articles.append(current)

    for idx, art in enumerate(articles):
        art["id"] = f"{date}_a{idx:02d}"
        art["position_in_issue"] = idx
        for p_idx, para in enumerate(art["paragraphs"]):
            para["id"] = f"{date}_a{idx:02d}_p{p_idx:02d}"
    issue["articles"] = articles
    (issue_dir / "ground_truth.json").write_text(json.dumps(issue, indent=2, ensure_ascii=False))
    print(f"{issue_dir}/ground_truth.json rebuilt from REVIEW.md: {len(articles)} units")


def main() -> None:
    commands = {"stitch": cmd_stitch, "assemble": cmd_assemble, "render": cmd_render, "ingest": cmd_ingest}
    if len(sys.argv) < 3 or sys.argv[1] not in ("bundle", *commands):
        print("usage: build_tentative_gt.py bundle <date> <page> | assemble|stitch|render|ingest <date>")
        raise SystemExit(1)
    if sys.argv[1] == "bundle":
        cmd_bundle(sys.argv[2], int(sys.argv[3]))
    else:
        commands[sys.argv[1]](sys.argv[2])


if __name__ == "__main__":
    main()
