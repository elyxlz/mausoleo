from __future__ import annotations

import json
import sys
import typing as tp

from mausoleo.eval.evaluate import article_text, normalize_text, text_overlap


def merge_predictions(
    primary: dict[str, tp.Any],
    secondary: dict[str, tp.Any],
    overlap_threshold: float = 0.3,
) -> dict[str, tp.Any]:
    primary_articles = primary.get("articles", [])
    secondary_articles = secondary.get("articles", [])

    primary_texts = [normalize_text(article_text(a)) for a in primary_articles]

    used_secondary: set[int] = set()
    for pi, pt in enumerate(primary_texts):
        for si, sa in enumerate(secondary_articles):
            if si in used_secondary:
                continue
            st = normalize_text(article_text(sa))
            if text_overlap(pt, st) >= overlap_threshold:
                used_secondary.add(si)

    new_articles: list[dict[str, tp.Any]] = []
    for si, sa in enumerate(secondary_articles):
        if si not in used_secondary:
            st = article_text(sa)
            if len(st.strip()) >= 30:
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
        print("Usage: ensemble_merge.py <primary_pred> <secondary_pred> <output>")
        sys.exit(1)

    primary = json.loads(open(sys.argv[1]).read())
    secondary = json.loads(open(sys.argv[2]).read())
    output_path = sys.argv[3]

    merged = merge_predictions(primary, secondary)

    primary_count = len(primary.get("articles", []))
    secondary_count = len(secondary.get("articles", []))
    merged_count = len(merged.get("articles", []))
    added = merged_count - primary_count

    print(f"Primary: {primary_count} articles")
    print(f"Secondary: {secondary_count} articles")
    print(f"Merged: {merged_count} articles (+{added} from secondary)")

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
