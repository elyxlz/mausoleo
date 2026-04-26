from __future__ import annotations

import json
import re
import sys
import typing as tp

from mausoleo.eval.evaluate import article_text, normalize_text, text_overlap


ITALIAN_ALPHA = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZàèéìòùÀÈÉÌÒÙ'")
PUNCT = set(".,;:!?()[]{}\"«»—–-\n")
WHITESPACE = set(" \t")


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    relevant = sum(1 for c in text if c in ITALIAN_ALPHA)
    total = sum(1 for c in text if c not in WHITESPACE)
    return relevant / max(total, 1)


def _repetition_penalty(text: str) -> float:
    run_pattern = re.compile(r"(.)\1{3,}")
    runs = run_pattern.findall(text)
    return 1.0 - min(len(runs) / max(len(text) // 50, 1), 1.0)


def _word_length_score(text: str) -> float:
    words = re.findall(r"[A-Za-zÀ-ÿ']+", text)
    if not words:
        return 0.0
    avg = sum(len(w) for w in words) / len(words)
    target = 5.2
    diff = abs(avg - target)
    return max(0.0, 1.0 - diff / 3.0)


def _single_char_token_ratio(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    singles = sum(1 for t in tokens if len(t) == 1 and t.isalpha())
    return singles / len(tokens)


def quality_score(text: str) -> float:
    if not text or len(text) < 10:
        return 0.0
    alpha = _alpha_ratio(text)
    rep = _repetition_penalty(text)
    wls = _word_length_score(text)
    singles = _single_char_token_ratio(text)
    return 0.4 * alpha + 0.2 * rep + 0.3 * wls + 0.1 * (1.0 - singles)


def _headline_quality(headline: str | None) -> float:
    if not headline:
        return 0.0
    words = re.findall(r"[A-Za-zÀ-ÿ']+", headline)
    if not words:
        return 0.0
    last = words[-1]
    vowels = set("aeiouàèéìòùAEIOUÀÈÉÌÒÙ")
    has_vowel = any(c in vowels for c in last)
    length_ok = 5 <= len(headline) <= 120
    alpha_ok = _alpha_ratio(headline) > 0.7
    no_trailing_dash = not headline.rstrip().endswith(("-", "'"))
    not_all_caps = not headline.isupper() or len(headline) <= 25
    score = (
        (0.25 if has_vowel else 0.0)
        + (0.25 if length_ok else 0.0)
        + (0.25 if alpha_ok else 0.0)
        + (0.15 if no_trailing_dash else 0.0)
        + (0.10 if not_all_caps else 0.0)
    )
    return score


def select_best_text(
    ensemble: dict[str, tp.Any],
    sources: list[dict[str, tp.Any]],
    match_threshold: float = 0.50,
    min_quality_delta: float = 0.15,
    headline_delta: float | None = None,
) -> dict[str, tp.Any]:
    if headline_delta is None:
        headline_delta = min_quality_delta
    ensemble_articles = ensemble.get("articles", [])
    source_normed = [
        [(i, normalize_text(article_text(a)), a) for i, a in enumerate(src.get("articles", []))]
        for src in sources
    ]

    replaced = 0
    headline_replaced = 0
    out_articles: list[dict[str, tp.Any]] = []
    for art in ensemble_articles:
        art_text = article_text(art)
        art_norm = normalize_text(art_text)
        art_quality = quality_score(art_text)
        art_headline = art.get("headline") or ""
        art_hl_quality = _headline_quality(art_headline)

        best_text = art_text
        best_quality = art_quality
        best_paragraphs: list[dict[str, str]] | None = None

        best_headline: str | None = art_headline or None
        best_hl_quality = art_hl_quality

        for src_idx, src_items in enumerate(source_normed):
            best_overlap = match_threshold
            best_match = None
            for _, cand_norm, cand_art in src_items:
                ov = text_overlap(art_norm, cand_norm)
                if ov > best_overlap:
                    best_overlap = ov
                    best_match = cand_art
            if best_match is None:
                continue
            cand_text = article_text(best_match)
            cand_quality = quality_score(cand_text)
            if cand_quality > best_quality + min_quality_delta:
                best_quality = cand_quality
                best_text = cand_text
                best_paragraphs = list(best_match.get("paragraphs", []))
            cand_headline = best_match.get("headline") or ""
            cand_hl_q = _headline_quality(cand_headline)
            if cand_hl_q > best_hl_quality + headline_delta:
                best_hl_quality = cand_hl_q
                best_headline = cand_headline or None

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
    print(f"  Replaced text for {replaced}/{len(ensemble_articles)} articles, headlines for {headline_replaced}")
    return out


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: quality_text_select.py <ensemble> <output> <source_1> [source_2 ...]")
        sys.exit(1)

    ensemble_path = sys.argv[1]
    output_path = sys.argv[2]
    source_paths = sys.argv[3:]

    ensemble = json.loads(open(ensemble_path).read())
    sources = [json.loads(open(p).read()) for p in source_paths]

    out = select_best_text(ensemble, sources)

    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
