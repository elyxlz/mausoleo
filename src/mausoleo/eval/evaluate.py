from __future__ import annotations

import dataclasses as dc
import re
import typing as tp


def compute_cer(reference: str, hypothesis: str) -> float:
    import jiwer

    if not reference:
        return 0.0 if not hypothesis else 1.0
    return jiwer.cer(reference, hypothesis)  # type: ignore[no-any-return]


def compute_wer(reference: str, hypothesis: str) -> float:
    import jiwer

    if not reference:
        return 0.0 if not hypothesis else 1.0
    return jiwer.wer(reference, hypothesis)  # type: ignore[no-any-return]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def text_overlap(a: str, b: str) -> float:
    a_words = set(normalize_text(a).split())
    b_words = set(normalize_text(b).split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words | b_words)


def article_text(article: dict[str, tp.Any]) -> str:
    return "\n".join(p.get("text", "") for p in article.get("paragraphs", []))


def article_pages(article: dict[str, tp.Any]) -> list[int]:
    return article.get("page_span", [])


@dc.dataclass(frozen=True)
class ArticleMatch:
    gt_index: int
    gt_headline: str
    gt_chars: int
    pred_index: int | None
    pred_headline: str | None
    cer: float
    wer: float
    headline_cer: float
    text_overlap: float
    page_span_correct: bool
    gt_pages: list[int]
    pred_pages: list[int]


@dc.dataclass(frozen=True)
class IssueResult:
    config: str
    date: str
    matches: list[ArticleMatch]
    article_precision: float
    article_recall: float
    article_f1: float
    article_gated_f1: float
    mean_cer: float
    mean_wer: float
    weighted_cer: float
    headline_cer: float
    full_text_cer: float
    full_text_wer: float
    page_accuracy: float
    ordering_score: float
    mausoleobench_score: float
    total_gt_articles: int
    total_pred_articles: int


def compute_ordering_score(matches: list[ArticleMatch]) -> float:
    paired = [(m.gt_index, m.pred_index) for m in matches if m.pred_index is not None]
    n = len(paired)
    if n < 2:
        return 0.0

    gt_ranks = sorted(range(n), key=lambda i: paired[i][0])
    pred_order = [paired[gt_ranks[i]][1] for i in range(n)]
    rank_map: dict[int, int] = {}
    for rank, val in enumerate(sorted(pred_order)):
        rank_map[val] = rank
    pred_ranks = [rank_map[v] for v in pred_order]

    sum_d_sq = sum((i - pred_ranks[i]) ** 2 for i in range(n))
    max_d_sq = n * (n * n - 1) / 3.0
    return max(0.0, 1.0 - sum_d_sq / max_d_sq)


def _unmatched(gi: int, gt_art: dict[str, tp.Any], gt_chars: int) -> ArticleMatch:
    return ArticleMatch(
        gt_index=gi,
        gt_headline=(gt_art.get("headline") or ""),
        gt_chars=gt_chars,
        pred_index=None,
        pred_headline=None,
        cer=1.0,
        wer=1.0,
        headline_cer=1.0,
        text_overlap=0.0,
        page_span_correct=False,
        gt_pages=article_pages(gt_art),
        pred_pages=[],
    )


def match_articles(
    gt_articles: list[dict[str, tp.Any]],
    pred_articles: list[dict[str, tp.Any]],
    overlap_threshold: float = 0.15,
) -> list[ArticleMatch]:
    gt_texts = [article_text(a) for a in gt_articles]
    pred_texts = [article_text(a) for a in pred_articles]
    gt_ws = [set(normalize_text(t).split()) for t in gt_texts]
    pred_ws = [set(normalize_text(t).split()) for t in pred_texts]
    gt_chars = [len(t.strip()) for t in gt_texts]

    # best-first global-greedy assignment over all candidate (gt, pred) pairs above threshold
    pairs: list[tuple[float, int, int]] = []
    for gi, gw in enumerate(gt_ws):
        if gt_chars[gi] < 20 or not gw:
            continue
        for pi, pw in enumerate(pred_ws):
            if not pw:
                continue
            ov = len(gw & pw) / len(gw | pw)
            if ov >= overlap_threshold:
                pairs.append((ov, gi, pi))
    pairs.sort(reverse=True)
    gt_to_pred: dict[int, tuple[int, float]] = {}
    used_pred: set[int] = set()
    for ov, gi, pi in pairs:
        if gi in gt_to_pred or pi in used_pred:
            continue
        gt_to_pred[gi] = (pi, ov)
        used_pred.add(pi)

    matches: list[ArticleMatch] = []
    for gi, gt_art in enumerate(gt_articles):
        if gi not in gt_to_pred:
            matches.append(_unmatched(gi, gt_art, gt_chars[gi]))
            continue
        pi, ov = gt_to_pred[gi]
        pred_art = pred_articles[pi]
        gt_norm = normalize_text(gt_texts[gi])
        pred_norm = normalize_text(pred_texts[pi])
        gt_h = normalize_text((gt_art.get("headline") or "").split("\n")[0])
        pred_h = normalize_text((pred_art.get("headline") or "").split("\n")[0])
        h_cer = compute_cer(gt_h, pred_h) if gt_h else 0.0
        matches.append(
            ArticleMatch(
                gt_index=gi,
                gt_headline=(gt_art.get("headline") or ""),
                gt_chars=gt_chars[gi],
                pred_index=pi,
                pred_headline=(pred_art.get("headline") or ""),
                cer=compute_cer(gt_norm, pred_norm),
                wer=compute_wer(gt_norm, pred_norm),
                headline_cer=h_cer,
                text_overlap=ov,
                page_span_correct=set(article_pages(gt_art)) == set(article_pages(pred_art)),
                gt_pages=article_pages(gt_art),
                pred_pages=article_pages(pred_art),
            )
        )
    return matches


def evaluate_issue(
    gt_issue: dict[str, tp.Any],
    pred_issue: dict[str, tp.Any],
    config: str = "",
    date: str = "",
) -> IssueResult:
    gt_articles = gt_issue.get("articles", [])
    pred_articles = pred_issue.get("articles", [])

    matches = match_articles(gt_articles, pred_articles)

    matched_gt = sum(1 for m in matches if m.pred_index is not None)
    matched_pred = len({m.pred_index for m in matches if m.pred_index is not None})

    precision = matched_pred / len(pred_articles) if pred_articles else 0.0
    recall = matched_gt / len(gt_articles) if gt_articles else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    matched = [m for m in matches if m.pred_index is not None]
    mean_cer = sum(m.cer for m in matched) / len(matched) if matched else 1.0
    mean_wer = sum(m.wer for m in matched) / len(matched) if matched else 1.0

    total_gt_chars = sum(m.gt_chars for m in matches)
    weighted_cer = sum(min(m.cer, 1.0) * m.gt_chars for m in matches) / total_gt_chars if total_gt_chars > 0 else 1.0

    # quality-gated segmentation: a match earns structure credit only if its text is mostly right.
    # q = max(0, 1 - cer/T) with T=2/3 -> text with >66% CER (more wrong than right) earns nothing.
    def quality(m: ArticleMatch) -> float:
        return max(0.0, 1.0 - 1.5 * min(m.cer, 1.0))

    gated = sum(quality(m) for m in matched)
    gated_recall = gated / len(gt_articles) if gt_articles else 0.0
    gated_precision = gated / len(pred_articles) if pred_articles else 0.0
    gated_f1 = 2 * gated_precision * gated_recall / (gated_precision + gated_recall) if (gated_precision + gated_recall) > 0 else 0.0

    # headline CER over ALL GT articles that HAVE a headline (unmatched -> 1.0, already stored)
    gt_with_headline = [m for m in matches if m.gt_headline]
    mean_headline_cer = sum(min(m.headline_cer, 1.0) for m in gt_with_headline) / len(gt_with_headline) if gt_with_headline else 1.0

    gt_full = " ".join(article_text(a) for a in gt_articles)
    pred_full = " ".join(article_text(a) for a in pred_articles)
    full_cer = compute_cer(normalize_text(gt_full), normalize_text(pred_full)) if gt_full.strip() else 1.0
    full_wer = compute_wer(normalize_text(gt_full), normalize_text(pred_full)) if gt_full.strip() else 1.0

    # page accuracy, quality-gated over all GT
    page_credit = sum(quality(m) for m in matched if m.page_span_correct)
    page_accuracy = page_credit / len(gt_articles) if gt_articles else 0.0

    # ordering only over good matches so scrambled/garbage text can't earn it
    ordering = compute_ordering_score([m for m in matched if m.cer <= 0.5])

    mausoleobench = 0.40 * (1.0 - weighted_cer) + 0.35 * gated_f1 + 0.05 * ordering + 0.10 * (1.0 - mean_headline_cer) + 0.10 * page_accuracy

    return IssueResult(
        config=config,
        date=date,
        matches=matches,
        article_precision=precision,
        article_recall=recall,
        article_f1=f1,
        mean_cer=mean_cer,
        mean_wer=mean_wer,
        weighted_cer=weighted_cer,
        headline_cer=mean_headline_cer,
        full_text_cer=full_cer,
        full_text_wer=full_wer,
        article_gated_f1=gated_f1,
        page_accuracy=page_accuracy,
        ordering_score=ordering,
        mausoleobench_score=mausoleobench,
        total_gt_articles=len(gt_articles),
        total_pred_articles=len(pred_articles),
    )
