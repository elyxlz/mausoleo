from __future__ import annotations

import json
import sys
import typing as tp


def _article_text(article: dict[str, tp.Any]) -> str:
    return "\n".join(p.get("text", "") for p in article.get("paragraphs", []))


def _ends_abruptly(text: str) -> bool:
    text = text.rstrip()
    if not text:
        return False
    last_char = text[-1]
    terminal_chars = set(".!?\"'»)]}")
    if last_char in terminal_chars:
        return False
    return True


def _starts_lowercase(text: str) -> bool:
    text = text.lstrip()
    if not text:
        return False
    first_char = text[0]
    return first_char.isalpha() and first_char.islower()


def stitch_cross_page(issue: dict[str, tp.Any]) -> dict[str, tp.Any]:
    articles = issue.get("articles", [])
    if not articles:
        return issue

    indexed = sorted(
        enumerate(articles),
        key=lambda x: (x[1].get("page_span", [999])[0] if x[1].get("page_span") else 999, x[0]),
    )
    ordered = [a for _, a in indexed]

    merge_ops: list[tuple[int, int]] = []
    for i in range(len(ordered) - 1):
        a = ordered[i]
        b = ordered[i + 1]
        pa = a.get("page_span", [])
        pb = b.get("page_span", [])
        if not pa or not pb:
            continue
        page_a_end = pa[-1]
        page_b_start = pb[0]
        if page_b_start != page_a_end + 1:
            continue

        text_a = _article_text(a)
        text_b = _article_text(b)
        if len(text_a) < 200 or len(text_b) < 200:
            continue
        if not _ends_abruptly(text_a):
            continue
        if not _starts_lowercase(text_b):
            continue
        if b.get("headline"):
            continue
        merge_ops.append((i, i + 1))

    if not merge_ops:
        return issue

    merged_indices = set()
    merged: list[dict[str, tp.Any]] = []
    for i, art in enumerate(ordered):
        if i in merged_indices:
            continue
        op_match = next((op for op in merge_ops if op[0] == i), None)
        if op_match:
            j = op_match[1]
            merged_indices.add(j)
            other = ordered[j]
            new_article = dict(art)
            new_paras = list(art.get("paragraphs", [])) + list(other.get("paragraphs", []))
            new_article["paragraphs"] = new_paras
            pa = art.get("page_span", [])
            pb = other.get("page_span", [])
            combined_pages = sorted(set(pa + pb))
            new_article["page_span"] = combined_pages
            merged.append(new_article)
        else:
            merged.append(art)

    result = dict(issue)
    result["articles"] = merged
    return result


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: stitch_cross_page.py <input> <output>")
        sys.exit(1)

    issue = json.loads(open(sys.argv[1]).read())
    before = len(issue.get("articles", []))
    stitched = stitch_cross_page(issue)
    after = len(stitched.get("articles", []))

    print(f"Before: {before} articles")
    print(f"After: {after} articles (stitched {before - after})")

    with open(sys.argv[2], "w") as f:
        json.dump(stitched, f, indent=2, ensure_ascii=False)
    print(f"Saved to {sys.argv[2]}")


if __name__ == "__main__":
    main()
