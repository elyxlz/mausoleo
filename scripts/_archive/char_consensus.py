"""Character-level consensus: align text from multiple sources, vote per char.

For each ensemble article, find 2-4 OTHER source predictions that match it
(text_overlap >= 0.45). Use difflib.SequenceMatcher to align them, then for each
position pick the character that appears most often across versions. Reject the
result if it differs too much from the original (>20% edit distance).

The hope: where vetturini's primary says "Taroni" and 3 other sources say
"Tarini" / "turini" / "Toroni" — character voting picks the consensus that
might be closer to the actual word. Won't fix systematic errors (where ALL
sources misread the same way) but should help when only the primary disagrees.
"""
from __future__ import annotations

import collections
import difflib
import json
import pathlib as pl
import re
import sys
import typing as tp


SOURCES_TO_USE = [
    "exp_045_qwen3vl_vllm",
    "col3_qwen3_8b_v2_structured",
    "col4_qwen3_8b_v2_structured",
    "exp_055_col6_ads_prompt",
    "exp_010_yolo_qwen3_8b",
    "exp_108_col3_qwen25vl",
    "exp_098_col5_qwen3vl_vllm",
    "exp_099_col2_qwen3vl_vllm",
    "exp_111_col2_qwen25vl",
    "yolo_qwen25_7b_v2_structured",
    "exp_052_col6_vllm",
    "exp_028_yolo_smallregion",
    "qwen3b_structured",
    "qwen_vl_3b_structured",
    "col3_qwen25_3b_v2_structured",
    "qwen25_3b_v2_structured",
]


def _article_text(article: dict[str, tp.Any]) -> str:
    return "\n".join(p.get("text", "") for p in article.get("paragraphs", []))


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _text_overlap(a: str, b: str) -> float:
    a_words = set(_normalize(a).split())
    b_words = set(_normalize(b).split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words | b_words)


def _find_matching_versions(target: dict[str, tp.Any], sources: list[dict[str, tp.Any]], min_overlap: float = 0.45, max_versions: int = 4) -> list[str]:
    target_text = _article_text(target)
    if len(target_text) < 200:
        return []
    target_norm = _normalize(target_text)
    matches: list[tuple[float, str]] = []
    for src in sources:
        for art in src.get("articles", []):
            text = _article_text(art)
            if abs(len(text) - len(target_text)) / max(len(target_text), 1) > 0.5:
                continue
            ov = _text_overlap(target_norm, _normalize(text))
            if ov >= min_overlap:
                matches.append((ov, text))
    matches.sort(key=lambda x: -x[0])
    seen: set[str] = set()
    result: list[str] = []
    for ov, t in matches:
        key = t[:100]
        if key in seen:
            continue
        seen.add(key)
        result.append(t)
        if len(result) >= max_versions:
            break
    return result


def _consensus_via_alignment(versions: list[str]) -> str:
    """Use the first (presumed best) version as the spine; for each region of difference,
    pick the character/word string that appears most often across other versions."""
    if not versions:
        return ""
    if len(versions) == 1:
        return versions[0]
    spine = versions[0]
    others = versions[1:]
    out_parts: list[str] = []
    spine_pos = 0
    consensus = list(spine)
    for other in others:
        sm = difflib.SequenceMatcher(None, spine, other, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
    # Simplified: when 3+ versions agree on a substitution against spine, apply it
    if len(others) < 2:
        return spine
    edit_proposals: dict[tuple[int, int], collections.Counter] = collections.defaultdict(collections.Counter)
    for other in others:
        sm = difflib.SequenceMatcher(None, spine, other, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            if i2 - i1 > 80 or j2 - j1 > 80:
                continue
            replacement = other[j1:j2]
            edit_proposals[(i1, i2)][replacement] += 1
    accepted: list[tuple[int, int, str]] = []
    for (i1, i2), counter in edit_proposals.items():
        replacement, count = counter.most_common(1)[0]
        if count >= max(2, len(others) // 2 + 1):
            accepted.append((i1, i2, replacement))
    accepted.sort(key=lambda x: x[0])
    if not accepted:
        return spine
    out: list[str] = []
    cursor = 0
    for i1, i2, repl in accepted:
        if i1 < cursor:
            continue
        out.append(spine[cursor:i1])
        out.append(repl)
        cursor = i2
    out.append(spine[cursor:])
    return "".join(out)


def consensus_correct(
    ensemble: dict[str, tp.Any],
    sources: list[dict[str, tp.Any]],
    max_edit_drift: float = 0.20,
) -> tuple[dict[str, tp.Any], int, int, int]:
    articles = list(ensemble.get("articles", []))
    new_articles = list(articles)
    accepted = 0
    rejected_drift = 0
    rejected_no_versions = 0
    for i, art in enumerate(articles):
        text = _article_text(art)
        if not (300 <= len(text) <= 4000):
            continue
        versions = [text] + _find_matching_versions(art, sources, max_versions=4)
        if len(versions) < 3:
            rejected_no_versions += 1
            continue
        consensus = _consensus_via_alignment(versions)
        if not consensus or consensus == text:
            continue
        sm = difflib.SequenceMatcher(None, text, consensus, autojunk=False)
        drift = 1.0 - sm.ratio()
        if drift > max_edit_drift:
            rejected_drift += 1
            continue
        new_a = dict(art)
        new_a["paragraphs"] = [{"text": consensus, "is_continuation": False}]
        new_articles[i] = new_a
        accepted += 1

    return {**ensemble, "articles": new_articles}, accepted, rejected_drift, rejected_no_versions


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: char_consensus.py <ens.json> <date> <out.json>")
        sys.exit(1)
    ens = json.loads(open(sys.argv[1]).read())
    date = sys.argv[2]
    out_path = sys.argv[3]
    cache = pl.Path("eval/predictions")
    sources = []
    for name in SOURCES_TO_USE:
        path = cache / f"{name}_{date}.json"
        if path.exists():
            sources.append(json.loads(path.read_text()))
    out, acc, rej_drift, rej_nv = consensus_correct(ens, sources)
    print(f"Accepted: {acc} | Rejected drift: {rej_drift} | Rejected too-few-versions: {rej_nv}")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
