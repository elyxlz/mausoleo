from __future__ import annotations

import json
import sys
import typing as tp

from mausoleo.eval.evaluate import article_text, normalize_text, text_overlap


def consensus_score(candidates: list[str]) -> list[float]:
    if len(candidates) < 2:
        return [0.0] * len(candidates)
    normed = [normalize_text(c) for c in candidates]
    scores = []
    for i, a in enumerate(normed):
        mean_ov = sum(text_overlap(a, b) for j, b in enumerate(normed) if j != i) / (len(normed) - 1)
        scores.append(mean_ov)
    return scores


def _find_best_match(
    target_norm: str,
    candidates: list[dict[str, tp.Any]],
    threshold: float = 0.50,
) -> dict[str, tp.Any] | None:
    best_ov = threshold
    best = None
    for cand in candidates:
        ov = text_overlap(target_norm, normalize_text(article_text(cand)))
        if ov > best_ov:
            best_ov = ov
            best = cand
    return best


def consensus_select(
    ensemble: dict[str, tp.Any],
    sources: list[dict[str, tp.Any]],
    match_threshold: float = 0.50,
    min_consensus_delta: float = 0.05,
    min_chars: int = 100,
) -> dict[str, tp.Any]:
    ensemble_articles = ensemble.get("articles", [])
    out_articles: list[dict[str, tp.Any]] = []
    replaced = 0
    for art in ensemble_articles:
        art_text_cur = article_text(art)
        if len(art_text_cur) < min_chars:
            out_articles.append(art)
            continue

        art_norm_cur = normalize_text(art_text_cur)
        candidates = [art]
        for src in sources:
            m = _find_best_match(art_norm_cur, src.get("articles", []), match_threshold)
            if m is not None and article_text(m) != art_text_cur:
                candidates.append(m)

        if len(candidates) < 3:
            out_articles.append(art)
            continue

        texts = [article_text(c) for c in candidates]
        scores = consensus_score(texts)
        ensemble_score = scores[0]
        best_idx = 0
        best_score = ensemble_score
        for i in range(1, len(candidates)):
            if scores[i] > best_score + min_consensus_delta:
                best_idx = i
                best_score = scores[i]

        if best_idx == 0:
            out_articles.append(art)
            continue

        chosen = candidates[best_idx]
        new_art = dict(art)
        new_art["paragraphs"] = [
            {"id": p.get("id", f"p{j}") if isinstance(p, dict) else f"p{j}",
             "text": p.get("text", str(p)) if isinstance(p, dict) else str(p)}
            for j, p in enumerate(chosen.get("paragraphs", []))
        ]
        out_articles.append(new_art)
        replaced += 1

    out = dict(ensemble)
    out["articles"] = out_articles
    print(f"  Consensus replaced text for {replaced}/{len(ensemble_articles)} articles")
    return out


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: consensus_text_select.py <ensemble> <output> <source_1> [source_2 ...]")
        sys.exit(1)

    ensemble = json.loads(open(sys.argv[1]).read())
    sources = [json.loads(open(p).read()) for p in sys.argv[3:]]

    out = consensus_select(ensemble, sources)

    with open(sys.argv[2], "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved to {sys.argv[2]}")


if __name__ == "__main__":
    main()
