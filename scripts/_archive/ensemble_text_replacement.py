from __future__ import annotations

import json
import sys
import typing as tp

from mausoleo.eval.evaluate import article_text, normalize_text, text_overlap


def merge_with_replacement(
    primary: dict[str, tp.Any],
    secondary: dict[str, tp.Any],
    overlap_threshold: float = 0.50,
    replace_ratio: float = 1.05,
    min_article_chars: int = 30,
) -> dict[str, tp.Any]:
    primary_articles = [dict(a) for a in primary.get("articles", [])]
    secondary_articles = secondary.get("articles", [])

    primary_texts = [normalize_text(article_text(a)) for a in primary_articles]
    secondary_texts = [normalize_text(article_text(a)) for a in secondary_articles]

    used_secondary: set[int] = set()
    for pi, pt in enumerate(primary_articles):
        p_norm = primary_texts[pi]
        p_len = len(article_text(pt))
        best_si = -1
        best_ov = 0.0
        for si, sa in enumerate(secondary_articles):
            if si in used_secondary:
                continue
            ov = text_overlap(p_norm, secondary_texts[si])
            if ov >= overlap_threshold and ov > best_ov:
                best_ov = ov
                best_si = si
        if best_si >= 0:
            used_secondary.add(best_si)
            s_len = len(article_text(secondary_articles[best_si]))
            if s_len >= p_len * replace_ratio:
                sa = dict(secondary_articles[best_si])
                sa["page_span"] = pt.get("page_span", sa.get("page_span"))
                if not sa.get("headline") and pt.get("headline"):
                    sa["headline"] = pt["headline"]
                primary_articles[pi] = sa

    new_articles: list[dict[str, tp.Any]] = []
    for si, sa in enumerate(secondary_articles):
        if si not in used_secondary and len(article_text(sa).strip()) >= min_article_chars:
            new_articles.append(sa)

    merged = list(primary_articles) + new_articles
    indexed = [(i, a) for i, a in enumerate(merged)]
    indexed.sort(key=lambda x: (x[1].get("page_span", [999])[0] if x[1].get("page_span") else 999, x[0]))
    merged = [a for _, a in indexed]

    result = dict(primary)
    result["articles"] = merged
    return result


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: ensemble_text_replacement.py <primary_pred> <secondary_pred> <output> [overlap=0.50] [replace_ratio=1.05]")
        sys.exit(1)

    primary = json.loads(open(sys.argv[1]).read())
    secondary = json.loads(open(sys.argv[2]).read())
    output_path = sys.argv[3]
    overlap = float(sys.argv[4]) if len(sys.argv) > 4 else 0.50
    ratio = float(sys.argv[5]) if len(sys.argv) > 5 else 1.05

    merged = merge_with_replacement(primary, secondary, overlap_threshold=overlap, replace_ratio=ratio)

    primary_count = len(primary.get("articles", []))
    secondary_count = len(secondary.get("articles", []))
    merged_count = len(merged.get("articles", []))
    added = merged_count - primary_count

    print(f"Primary: {primary_count} articles")
    print(f"Secondary: {secondary_count} articles")
    print(f"Merged: {merged_count} articles (+{added} from secondary, overlap={overlap}, ratio={ratio})")

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
