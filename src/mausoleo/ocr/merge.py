from __future__ import annotations

import difflib
import re
import typing as tp

from mausoleo.eval.evaluate import article_text, normalize_text, text_overlap


REPEAT_PATTERN = re.compile(r"(\s*[\.\-\=\_\*])\s*((?:[\.\-\=\_\*]\s*){6,})")
ITALIAN_ALPHA = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZàèéìòùÀÈÉÌÒÙ'")
WHITESPACE = set(" \t")


def trim_trailing_garbage(text: str) -> str:
    if not text:
        return text
    trimmed = text
    trimmed = re.sub(r"(\s*\.\s*){5,}\s*$", ".", trimmed)
    trimmed = re.sub(r"(\s*\-\s*){5,}\s*$", "-", trimmed)
    trimmed = re.sub(r"(\s*\*\s*){5,}\s*$", "", trimmed)
    trimmed = re.sub(r"(\s*\"\s*){3,}\s*$", "", trimmed)
    words = trimmed.rstrip().split()
    if words and all(len(w) <= 2 for w in words[-30:]):
        while words and len(words[-1]) <= 2 and words[-1] in {".", "-", "_", "=", "*", ",", ":", ";"}:
            words.pop()
        trimmed = " ".join(words)
    return trimmed


def looks_like_json_blob(text: str) -> bool:
    if not text:
        return False
    first200 = text[:200].strip()
    if first200.startswith("```"):
        first200 = first200.lstrip("`").lstrip("json").lstrip("JSON").strip()
    if first200.startswith("{") and '"articles"' in first200:
        return True
    if first200.startswith("{") and ('"unit_type"' in first200 or '"paragraphs"' in first200):
        return True
    if first200.startswith("[") and "{" in first200 and '"text"' in first200:
        return True
    if '"unit_type"' in text[:500] and '"paragraphs"' in text[:500]:
        return True
    return False


def trim_predictions(pred: dict[str, tp.Any]) -> dict[str, tp.Any]:
    articles = list(pred.get("articles", []))
    trimmed_count = 0
    dropped_count = 0
    new_articles = []
    for art in articles:
        paragraphs = list(art.get("paragraphs", []))
        if not paragraphs:
            new_articles.append(art)
            continue
        combined = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in paragraphs)
        if looks_like_json_blob(combined):
            dropped_count += 1
            continue
        new_paragraphs = []
        changed = False
        for p in paragraphs:
            if not isinstance(p, dict):
                new_paragraphs.append({"text": str(p)})
                continue
            t = p.get("text", "")
            t_new = trim_trailing_garbage(t)
            if t_new != t:
                changed = True
            new_paragraphs.append({**p, "text": t_new})
        if changed:
            trimmed_count += 1
        new_art = dict(art)
        new_art["paragraphs"] = new_paragraphs
        new_articles.append(new_art)
    out = dict(pred)
    out["articles"] = new_articles
    print(f"  Trimmed {trimmed_count} articles, dropped {dropped_count} JSON blobs ({len(articles)} -> {len(new_articles)})")
    return out


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

        for src_items in source_normed:
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


def _norm_loose(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _ends_unfinished(text: str) -> bool:
    text = text.rstrip()
    if not text:
        return False
    return text[-1] not in ".!?»\"')]}…"


def _starts_match(a: str, b: str, min_chars: int = 60, min_ratio: float = 0.85) -> bool:
    a_start = _norm_loose(a)[:min_chars]
    b_start = _norm_loose(b)[:min_chars]
    if len(a_start) < min_chars or len(b_start) < min_chars:
        return False
    return difflib.SequenceMatcher(None, a_start, b_start, autojunk=False).ratio() >= min_ratio


def _is_continuation_article(article: dict[str, tp.Any]) -> bool:
    text = "\n".join(p.get("text", "") for p in article.get("paragraphs", [])).lstrip()
    if not text:
        return False
    headline = article.get("headline") or ""
    if len(headline) > 20:
        return False
    first = text[0]
    if not first.isalpha():
        return False
    if not first.islower():
        return False
    return True


def _crosspage_pairs(col1: dict[str, tp.Any], max_ahead: int = 12) -> list[tuple[str, str]]:
    arts = col1.get("articles", [])
    pairs: list[tuple[str, str]] = []
    for i in range(len(arts) - 1):
        head_text = "\n".join(p.get("text", "") for p in arts[i].get("paragraphs", []))
        if len(head_text) < 300:
            continue
        if not _ends_unfinished(head_text):
            continue
        for j in range(i + 1, min(i + 1 + max_ahead, len(arts))):
            cont = arts[j]
            if not _is_continuation_article(cont):
                continue
            cont_text = "\n".join(p.get("text", "") for p in cont.get("paragraphs", []))
            if len(cont_text) < 200:
                continue
            pairs.append((head_text, cont_text))
            break
    return pairs


def replace_with_pairs(
    ensemble: dict[str, tp.Any],
    col1_predictions: list[dict[str, tp.Any]],
) -> tuple[dict[str, tp.Any], int, int]:
    pairs: list[tuple[str, str]] = []
    for col1 in col1_predictions:
        pairs.extend(_crosspage_pairs(col1))

    articles = list(ensemble.get("articles", []))
    new_articles = list(articles)
    replaced = 0
    for ai, art in enumerate(articles):
        text = "\n".join(p.get("text", "") for p in art.get("paragraphs", []))
        if len(text) < 400:
            continue
        if not _ends_unfinished(text):
            continue
        original_pages = art.get("page_span") or []
        if len(original_pages) > 1:
            continue
        best_combined: str | None = None
        best_total_len = 0
        for head_text, cont_text in pairs:
            if not _starts_match(text, head_text):
                continue
            combined = head_text.rstrip() + " " + cont_text.lstrip()
            if len(combined) < len(text) * 1.15:
                continue
            if len(combined) > best_total_len:
                best_total_len = len(combined)
                best_combined = combined
        if best_combined is None:
            continue
        new_art = dict(art)
        new_art["paragraphs"] = [{"text": best_combined, "is_continuation": False}]
        if original_pages:
            new_art["page_span"] = sorted(set(original_pages + [original_pages[-1] + 1]))
        new_articles[ai] = new_art
        replaced += 1

    return {**ensemble, "articles": new_articles}, len(pairs), replaced
