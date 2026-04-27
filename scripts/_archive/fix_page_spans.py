from __future__ import annotations

import json
import sys
import typing as tp

from mausoleo.eval.evaluate import article_text, normalize_text, text_overlap


def _find_best_match(
    target_norm: str,
    candidates_norm: list[str],
    threshold: float = 0.50,
) -> int:
    best_i = -1
    best_ov = threshold
    for i, cn in enumerate(candidates_norm):
        ov = text_overlap(target_norm, cn)
        if ov > best_ov:
            best_ov = ov
            best_i = i
    return best_i


def union_page_spans(
    ensemble_path: str,
    source_paths: list[str],
    output_path: str,
    overlap_threshold: float = 0.50,
) -> None:
    ensemble = json.loads(open(ensemble_path).read())
    sources = [json.loads(open(p).read()) for p in source_paths]

    source_normed = [
        [normalize_text(article_text(a)) for a in s.get("articles", [])]
        for s in sources
    ]

    fixed_count = 0
    out_articles = []
    for art in ensemble.get("articles", []):
        art_norm = normalize_text(article_text(art))
        all_pages: set[int] = set(art.get("page_span", []))
        start_pages = set(all_pages)

        for si, src in enumerate(sources):
            best = _find_best_match(art_norm, source_normed[si], overlap_threshold)
            if best >= 0:
                src_pages = src["articles"][best].get("page_span", [])
                all_pages.update(src_pages)

        new_pages = sorted(p for p in all_pages if p > 0)
        if set(new_pages) != start_pages and new_pages:
            fixed_count += 1
            new_art = dict(art)
            new_art["page_span"] = new_pages
            out_articles.append(new_art)
        else:
            out_articles.append(art)

    out = dict(ensemble)
    out["articles"] = out_articles

    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Fixed page_span for {fixed_count}/{len(out_articles)} articles")
    print(f"Saved to {output_path}")


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: fix_page_spans.py <ensemble_pred> <output> <source_pred_1> [source_pred_2 ...]")
        sys.exit(1)

    ensemble_path = sys.argv[1]
    output_path = sys.argv[2]
    source_paths = sys.argv[3:]

    union_page_spans(ensemble_path, source_paths, output_path)


if __name__ == "__main__":
    main()
