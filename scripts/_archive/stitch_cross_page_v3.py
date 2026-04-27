from __future__ import annotations

import json
import re
import sys
import typing as tp


def _article_text(article: dict[str, tp.Any]) -> str:
    return "\n".join(p.get("text", "") for p in article.get("paragraphs", []))


def _last_word(text: str) -> str:
    words = re.findall(r"[A-Za-zÀ-ÿ']+", text)
    return words[-1] if words else ""


def _first_chars(text: str, n: int = 50) -> str:
    return text.lstrip()[:n]


def _has_vowel(word: str) -> bool:
    return bool(re.search(r"[aeiouàèéìòùAEIOUÀÈÉÌÒÙ]", word))


def _is_obvious_truncated_word(word: str) -> bool:
    if len(word) < 3:
        return False
    if len(word) >= 3 and not _has_vowel(word):
        return True
    common_endings = {"e", "o", "a", "i", "he", "to", "re", "ta", "ne", "ri", "ti", "mo"}
    if len(word) >= 4 and word[-2:].lower() in common_endings:
        return False
    return False


def _ends_with_obvious_truncation(text: str) -> bool:
    text = text.rstrip()
    if not text:
        return False
    if text[-1] in ".!?»\")]}":
        return False
    if text.endswith("—") or text.endswith("…"):
        return True
    return False


def _headline_truncated_midword(headline: str | None) -> bool:
    if not headline:
        return False
    words = re.findall(r"[A-Za-zÀ-ÿ']+", headline)
    if len(words) < 1:
        return False
    last = words[-1]
    if len(last) <= 2:
        return False
    if len(last) >= 3 and not _has_vowel(last):
        return True
    if headline.rstrip().endswith("-"):
        return True
    return False


def _starts_lowercase_midword(text: str) -> bool:
    text = text.lstrip()
    if not text:
        return False
    first = text[0]
    if not first.isalpha():
        return False
    if not first.islower():
        return False
    first_word = re.match(r"[A-Za-zÀ-ÿ']+", text)
    if not first_word:
        return False
    fw = first_word.group(0)
    if len(fw) >= 3 and not _has_vowel(fw):
        return True
    return False


def stitch_articles(issue: dict[str, tp.Any]) -> tuple[dict[str, tp.Any], int]:
    articles = issue.get("articles", [])
    if not articles:
        return issue, 0

    ordered_idx = sorted(
        range(len(articles)),
        key=lambda i: (
            articles[i].get("page_span", [999])[0] if articles[i].get("page_span") else 999,
            i,
        ),
    )
    articles_sorted = [articles[i] for i in ordered_idx]

    pairs: list[tuple[int, int]] = []
    consumed: set[int] = set()

    for i in range(len(articles_sorted)):
        if i in consumed:
            continue
        a = articles_sorted[i]
        pa = a.get("page_span", [])
        if not pa:
            continue
        text_a = _article_text(a)
        if len(text_a) < 800:
            continue
        last_page_a = pa[-1]

        a_headline_truncated = _headline_truncated_midword(a.get("headline"))
        a_body_truncated = _ends_with_obvious_truncation(text_a)
        if not (a_headline_truncated or a_body_truncated):
            continue

        for j in range(i + 1, len(articles_sorted)):
            if j in consumed:
                continue
            b = articles_sorted[j]
            pb = b.get("page_span", [])
            if not pb:
                continue
            if pb[0] > last_page_a + 1:
                break
            if pb[0] != last_page_a + 1:
                continue
            text_b = _article_text(b)
            if len(text_b) < 400:
                continue
            b_starts_midword = _starts_lowercase_midword(text_b)
            b_headline_truncated = _headline_truncated_midword(b.get("headline"))
            if not (b_starts_midword or b_headline_truncated):
                continue
            pairs.append((i, j))
            consumed.add(j)
            break

    if not pairs:
        return issue, 0

    merge_map = {i: j for i, j in pairs}
    result_articles: list[dict[str, tp.Any]] = []
    for i, art in enumerate(articles_sorted):
        if i in consumed:
            continue
        if i in merge_map:
            j = merge_map[i]
            successor = articles_sorted[j]
            new_art = dict(art)
            new_paras = list(art.get("paragraphs", [])) + list(successor.get("paragraphs", []))
            new_art["paragraphs"] = new_paras
            pa = art.get("page_span", [])
            pb = successor.get("page_span", [])
            new_art["page_span"] = sorted(set(pa + pb))
            result_articles.append(new_art)
        else:
            result_articles.append(art)

    out = dict(issue)
    out["articles"] = result_articles
    return out, len(consumed)


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: stitch_cross_page_v3.py <input> <output>")
        sys.exit(1)

    issue = json.loads(open(sys.argv[1]).read())
    before = len(issue.get("articles", []))
    out, stitched = stitch_articles(issue)
    after = len(out.get("articles", []))

    print(f"Before: {before} | After: {after} | Stitched: {stitched}")

    with open(sys.argv[2], "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
