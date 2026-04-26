from __future__ import annotations

import json
import re
import sys
import typing as tp


TERMINAL_CHARS = set(".!?»\")'])}")
ITALIAN_COMMON_WORDS = {
    "il", "la", "lo", "i", "gli", "le", "un", "una", "di", "del", "della", "dei", "degli", "delle",
    "che", "e", "ed", "o", "od", "a", "al", "alla", "allo", "ai", "agli", "alle", "da", "dal",
    "dalla", "con", "su", "sul", "sulla", "per", "in", "nel", "nella", "tra", "fra", "non",
    "ma", "se", "si", "suo", "sua", "suoi", "sue", "mio", "mia", "loro", "questo", "questa",
    "sono", "era", "sia", "erano", "fu", "furono", "ha", "hanno", "aveva",
}


def _article_text(article: dict[str, tp.Any]) -> str:
    return "\n".join(p.get("text", "") for p in article.get("paragraphs", []))


def _article_chars(article: dict[str, tp.Any]) -> int:
    return len(_article_text(article))


def _last_word(text: str) -> str:
    words = re.findall(r"[A-Za-zÀ-ÿ']+", text)
    return words[-1] if words else ""


def _first_word(text: str) -> str:
    m = re.search(r"[A-Za-zÀ-ÿ']+", text)
    return m.group(0) if m else ""


def _looks_like_partial_word(word: str) -> bool:
    if len(word) < 3:
        return False
    vowels = set("aeiouàèéìòùAEIOUÀÈÉÌÒÙ")
    if len(word) >= 3 and not any(c in vowels for c in word):
        return True
    if word.endswith("-"):
        return True
    return False


def _ends_clearly_truncated(text: str) -> bool:
    text = text.rstrip()
    if not text:
        return False
    last = text[-1]
    if last in TERMINAL_CHARS:
        return False
    if text.endswith("—") or text.endswith("…") or text.endswith("-"):
        return True
    last = _last_word(text)
    if not last:
        return False
    if last.lower() in ITALIAN_COMMON_WORDS:
        return True
    return False


def _starts_as_fragment(text: str) -> bool:
    text = text.lstrip()
    if not text:
        return False
    first = text[0]
    if not first.isalpha():
        return False
    if first.islower():
        return True
    fw = _first_word(text)
    if fw and fw.lower() in {"che", "di", "del", "della", "con", "per", "ma", "se", "si", "non"}:
        return True
    return False


def _headline_looks_truncated(headline: str | None) -> bool:
    if not headline:
        return False
    words = re.findall(r"[A-Za-zÀ-ÿ']+", headline)
    if not words:
        return False
    last = words[-1]
    if _looks_like_partial_word(last):
        return True
    if len(last) <= 3 and last.isalpha() and last.lower() not in ITALIAN_COMMON_WORDS:
        return True
    return False


def stitch_articles(issue: dict[str, tp.Any]) -> tuple[dict[str, tp.Any], int]:
    articles = issue.get("articles", [])
    if not articles:
        return issue, 0

    ordered = sorted(
        range(len(articles)),
        key=lambda i: (
            articles[i].get("page_span", [999])[0] if articles[i].get("page_span") else 999,
            i,
        ),
    )
    articles_sorted = [articles[i] for i in ordered]

    pairs: list[tuple[int, int]] = []
    consumed: set[int] = set()

    for i in range(len(articles_sorted)):
        if i in consumed:
            continue
        a = articles_sorted[i]
        pa = a.get("page_span", [])
        if not pa or _article_chars(a) < 500:
            continue
        last_page_a = pa[-1]
        text_a = _article_text(a)
        a_truncates_body = _ends_clearly_truncated(text_a)
        a_truncated_headline = _headline_looks_truncated(a.get("headline"))
        if not (a_truncates_body or a_truncated_headline):
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
            if _article_chars(b) < 200:
                continue
            text_b = _article_text(b)
            b_fragment = _starts_as_fragment(text_b) or _headline_looks_truncated(b.get("headline"))
            if not b_fragment:
                continue
            pairs.append((i, j))
            consumed.add(j)
            break

    result_articles: list[dict[str, tp.Any]] = []
    merge_map = {i: j for i, j in pairs}
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
        print("Usage: stitch_cross_page_v2.py <input> <output>")
        sys.exit(1)

    issue = json.loads(open(sys.argv[1]).read())
    before = len(issue.get("articles", []))
    out, stitched = stitch_articles(issue)
    after = len(out.get("articles", []))

    print(f"Before: {before} | After: {after} | Stitched: {stitched}")

    with open(sys.argv[2], "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved to {sys.argv[2]}")


if __name__ == "__main__":
    main()
