from __future__ import annotations

import json
import sys
import typing as tp

from mausoleo.eval.evaluate import article_text, normalize_text, text_overlap
from quality_text_select import quality_score, _headline_quality


def length_quality(text: str) -> float:
    q = quality_score(text)
    if q < 0.30:
        return 0.0
    import math
    return q * math.sqrt(max(len(text), 1))


def select_longest_quality_text(
    ensemble: dict[str, tp.Any],
    sources: list[dict[str, tp.Any]],
    match_threshold: float = 0.35,
    min_quality: float = 0.40,
    headline_delta: float = 0.15,
) -> dict[str, tp.Any]:
    ensemble_articles = ensemble.get("articles", [])
    source_normed = [
        [(normalize_text(article_text(a)), a) for a in src.get("articles", [])]
        for src in sources
    ]

    replaced = 0
    headline_replaced = 0
    out_articles: list[dict[str, tp.Any]] = []

    for art in ensemble_articles:
        art_text = article_text(art)
        art_norm = normalize_text(art_text)
        art_score = length_quality(art_text)
        art_headline = art.get("headline") or ""
        art_hl_q = _headline_quality(art_headline)

        best_text = art_text
        best_score = art_score
        best_paragraphs: list[tp.Any] | None = None
        best_headline = art_headline or None
        best_hl_q = art_hl_q

        for src_items in source_normed:
            best_ov = match_threshold
            best_match = None
            for cand_norm, cand_art in src_items:
                ov = text_overlap(art_norm, cand_norm)
                if ov > best_ov:
                    best_ov = ov
                    best_match = cand_art
            if best_match is None:
                continue
            cand_text = article_text(best_match)
            cand_q = quality_score(cand_text)
            if cand_q < min_quality:
                continue
            cand_score = length_quality(cand_text)
            if cand_score > best_score * 1.10:
                best_score = cand_score
                best_text = cand_text
                best_paragraphs = list(best_match.get("paragraphs", []))
            cand_hl_q = _headline_quality(best_match.get("headline") or "")
            if cand_hl_q > best_hl_q + headline_delta:
                best_hl_q = cand_hl_q
                best_headline = best_match.get("headline") or None

        new_art = dict(art)
        if best_paragraphs is not None and best_text != art_text:
            new_art["paragraphs"] = [
                {"id": p.get("id", f"p{j}"), "text": p.get("text", str(p)) if isinstance(p, dict) else str(p)}
                for j, p in enumerate(best_paragraphs)
            ]
            replaced += 1
        if best_headline != (art.get("headline") or None):
            new_art["headline"] = best_headline
            headline_replaced += 1
        out_articles.append(new_art)

    out = dict(ensemble)
    out["articles"] = out_articles
    print(f"  Length-aware: replaced text for {replaced}/{len(ensemble_articles)}, headlines for {headline_replaced}")
    return out


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: length_aware_select.py <ensemble> <output> <source_1> [source_2 ...]")
        sys.exit(1)
    ensemble = json.loads(open(sys.argv[1]).read())
    sources = [json.loads(open(p).read()) for p in sys.argv[3:]]
    out = select_longest_quality_text(ensemble, sources)
    with open(sys.argv[2], "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
