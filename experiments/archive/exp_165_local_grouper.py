from __future__ import annotations

import json
import pathlib as pl
import re
import sys
import time
import typing as tp

sys.path.insert(0, "src")

EXP_NAME = pl.Path(__file__).stem
GROUND_TRUTH_DIR = pl.Path("eval/ground_truth")
PREDICTIONS_DIR = pl.Path("eval/predictions")
WORK_DIR = pl.Path("eval/autoresearch/semgroup")
GROUPER_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

PROMPT_HEADER = (
    "You segment historical Italian newspaper pages into content units. Below is the ordered list of OCR regions "
    "for ONE page: idx | class (title/text) | y-range | text excerpt.\n"
    "Group ALL region idxs into printed units (article, advertisement, notice, obituary, editorial, other):\n"
    "- a unit is usually a title region plus the text regions of its body flowing down the column\n"
    "- consecutive title regions at a unit start form one headline block\n"
    "- distinct small ads/notices are separate units even without titles (use the text to find boundaries)\n"
    "- long articles may have internal crossheads: keep them ONE unit\n"
    "- every idx appears in exactly one unit\n"
    'Output ONLY a JSON array: [{"unit_type": "...", "headline_region": <idx or null>, "regions": [idx, ...]}, ...] '
    "in reading order. No prose, no markdown fences.\n\nREGIONS:\n"
)

_ENGINE: dict[str, tp.Any] = {}


def _get_llm() -> tp.Any:
    if "llm" not in _ENGINE:
        from vllm import LLM

        _ENGINE["llm"] = LLM(
            model=GROUPER_MODEL,
            trust_remote_code=True,
            gpu_memory_utilization=0.92,
            max_model_len=16384,
            enforce_eager=False,
            dtype="bfloat16",
            seed=0,
        )
    return _ENGINE["llm"]


def _page_prompt(regions: list[dict[str, tp.Any]]) -> str:
    lines = []
    for r in regions:
        text = " ".join(r["text"].split())[:100]
        lines.append(f"{r['idx']} | {r['class']} | y{r['bbox'][1]}-{r['bbox'][3]} | {text}")
    return PROMPT_HEADER + "\n".join(lines) + "\n\nJSON:"


def _parse_groups(raw: str) -> list[dict[str, tp.Any]] | None:
    raw = raw.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.M).strip()
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    groups = []
    for g in parsed:
        if isinstance(g, dict) and isinstance(g.get("regions"), list):
            groups.append(g)
    return groups or None


def _fallback_groups(regions: list[dict[str, tp.Any]]) -> list[dict[str, tp.Any]]:
    groups: list[dict[str, tp.Any]] = []
    current: dict[str, tp.Any] | None = None
    for r in regions:
        if r["class"] == "title":
            if (
                current is not None
                and current["headline_region"] is not None
                and len(current["regions"]) == sum(1 for i in current["regions"] if i in current.get("_titles", set()))
            ):
                current["regions"].append(r["idx"])
                current.setdefault("_titles", set()).add(r["idx"])
                continue
            if current is not None:
                groups.append(current)
            current = {"unit_type": "article", "headline_region": r["idx"], "regions": [r["idx"]], "_titles": {r["idx"]}}
        else:
            if current is None:
                current = {"unit_type": "article", "headline_region": None, "regions": [], "_titles": set()}
            current["regions"].append(r["idx"])
    if current is not None:
        groups.append(current)
    for g in groups:
        g.pop("_titles", None)
    return groups


def _repair_coverage(groups: list[dict[str, tp.Any]], regions: list[dict[str, tp.Any]]) -> list[dict[str, tp.Any]]:
    valid = {r["idx"] for r in regions}
    seen: set[int] = set()
    for g in groups:
        g["regions"] = [i for i in g["regions"] if i in valid and i not in seen]
        seen.update(g["regions"])
    groups = [g for g in groups if g["regions"]]
    orphans = sorted(valid - seen)
    order = {r["idx"]: pos for pos, r in enumerate(regions)}
    for orphan in orphans:
        best, best_dist = None, None
        for g in groups:
            dist = min(abs(order[orphan] - order[i]) for i in g["regions"])
            if best_dist is None or dist < best_dist:
                best, best_dist = g, dist
        if best is None:
            groups.append({"unit_type": "article", "headline_region": None, "regions": [orphan]})
        else:
            best["regions"] = sorted({*best["regions"], orphan}, key=lambda i: order[i])
    return groups


def cmd_group(date: str) -> None:
    from vllm import SamplingParams

    regions = json.loads((WORK_DIR / f"regions_{date}.json").read_text())
    by_page: dict[int, list[dict[str, tp.Any]]] = {}
    for r in regions:
        by_page.setdefault(r["page"], []).append(r)

    llm = _get_llm()
    t0 = time.time()
    prompts = [_page_prompt(by_page[p]) for p in sorted(by_page)]
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=6144))
    all_groups: list[dict[str, tp.Any]] = []
    fallbacks = 0
    for pos, (page, out) in enumerate(zip(sorted(by_page), outputs)):
        parsed = _parse_groups(out.outputs[0].text)
        if parsed is None:
            retry = llm.generate([prompts[pos]], SamplingParams(temperature=0.2, max_tokens=6144, seed=1))
            parsed = _parse_groups(retry[0].outputs[0].text)
        if parsed is None:
            parsed = _fallback_groups(by_page[page])
            fallbacks += 1
        all_groups.extend(_repair_coverage(parsed, by_page[page]))
    elapsed = time.time() - t0
    out_path = WORK_DIR / f"groups_local_{date}.json"
    out_path.write_text(json.dumps(all_groups, indent=1))
    n_pages = len(by_page)
    print(
        f"{date}: {len(all_groups)} groups, {fallbacks} fallback pages | grouping {elapsed:.1f}s = {elapsed / n_pages:.2f} s/page -> {out_path}"
    )


def cmd_assemble(date: str) -> None:
    from mausoleo.ocr.operators.merge import squeeze_char_runs

    regions = {r["idx"]: r for r in json.loads((WORK_DIR / f"regions_{date}.json").read_text())}
    groups = json.loads((WORK_DIR / f"groups_local_{date}.json").read_text())
    pages = sorted(GROUND_TRUTH_DIR.joinpath(date).glob("*.jpeg"), key=lambda p: int(p.stem))

    articles: list[dict[str, tp.Any]] = []
    used: set[int] = set()
    for group in groups:
        idxs = [i for i in group["regions"] if i in regions and i not in used]
        if not idxs:
            continue
        used.update(idxs)
        head_parts: list[str] = []
        body_idxs: list[int] = []
        in_head = True
        for i in idxs:
            if in_head and regions[i]["class"] == "title":
                head_parts.append(" ".join(regions[i]["text"].split()))
            else:
                in_head = False
                body_idxs.append(i)
        headline = "\n".join(p for p in head_parts if p)[:300] or None
        paragraphs = [{"text": squeeze_char_runs(regions[i]["text"]).strip()} for i in body_idxs]
        paragraphs = [p for p in paragraphs if p["text"]]
        if not paragraphs and not headline:
            continue
        if not paragraphs:
            paragraphs = [{"text": headline}]
        page_span = sorted({regions[i]["page"] for i in idxs})
        articles.append(
            {"unit_type": group.get("unit_type", "article"), "headline": headline, "paragraphs": paragraphs, "page_span": page_span}
        )

    for idx, art in enumerate(articles):
        art["id"] = f"{date}_a{idx:02d}"
        art["position_in_issue"] = idx
        for p_idx, para in enumerate(art["paragraphs"]):
            para["id"] = f"{date}_a{idx:02d}_p{p_idx:02d}"
    issue = {"date": date, "source": "il_messaggero", "page_count": len(pages), "articles": articles}
    out = PREDICTIONS_DIR / f"{EXP_NAME}_{date}.json"
    out.write_text(json.dumps(issue, indent=2, ensure_ascii=False))
    print(f"{date}: {len(articles)} articles -> {out}")


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in ("group", "assemble"):
        raise SystemExit(f"usage: {EXP_NAME}.py group <date...> | assemble <date...>")
    for date in sys.argv[2:]:
        (cmd_group if sys.argv[1] == "group" else cmd_assemble)(date)


if __name__ == "__main__":
    main()
