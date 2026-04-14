from __future__ import annotations

import dataclasses as dc
import re
import typing as tp

from mausoleo.eval.metrics import compute_cer, compute_wer


@dc.dataclass(frozen=True)
class ArticleMatch:
    gt_index: int
    gt_headline: str
    pred_index: int | None
    pred_headline: str | None
    cer: float
    wer: float
    text_overlap: float
    page_span_correct: bool
    gt_pages: list[int]
    pred_pages: list[int]


@dc.dataclass(frozen=True)
class EvalResult:
    matches: list[ArticleMatch]
    article_precision: float
    article_recall: float
    article_f1: float
    mean_cer: float
    mean_wer: float
    page_accuracy: float
    total_gt_articles: int
    total_pred_articles: int


def _normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _text_overlap(a: str, b: str) -> float:
    a_norm = _normalize(a)
    b_norm = _normalize(b)
    if not a_norm or not b_norm:
        return 0.0
    a_words = set(a_norm.split())
    b_words = set(b_norm.split())
    if not a_words or not b_words:
        return 0.0
    intersection = len(a_words & b_words)
    union = len(a_words | b_words)
    return intersection / union


def _extract_articles(issue: dict[str, tp.Any]) -> list[dict[str, tp.Any]]:
    return issue.get("articles", [])


def _article_text(article: dict[str, tp.Any]) -> str:
    parts = []
    for p in article.get("paragraphs", []):
        parts.append(p.get("text", ""))
    return "\n".join(parts)


def _article_pages(article: dict[str, tp.Any]) -> list[int]:
    return article.get("page_span", [])


def match_articles(
    gt_articles: list[dict[str, tp.Any]],
    pred_articles: list[dict[str, tp.Any]],
    overlap_threshold: float = 0.15,
) -> list[ArticleMatch]:
    gt_texts = [_article_text(a) for a in gt_articles]
    pred_texts = [_article_text(a) for a in pred_articles]

    used_pred: set[int] = set()
    matches: list[ArticleMatch] = []

    for gi, gt_art in enumerate(gt_articles):
        gt_text = gt_texts[gi]
        if len(gt_text.strip()) < 20:
            matches.append(ArticleMatch(
                gt_index=gi,
                gt_headline=gt_art.get("headline", ""),
                pred_index=None,
                pred_headline=None,
                cer=1.0,
                wer=1.0,
                text_overlap=0.0,
                page_span_correct=False,
                gt_pages=_article_pages(gt_art),
                pred_pages=[],
            ))
            continue

        best_pi = -1
        best_overlap = 0.0
        for pi, pred_text in enumerate(pred_texts):
            if pi in used_pred:
                continue
            overlap = _text_overlap(gt_text, pred_text)
            if overlap > best_overlap:
                best_overlap = overlap
                best_pi = pi

        if best_pi >= 0 and best_overlap >= overlap_threshold:
            used_pred.add(best_pi)
            pred_art = pred_articles[best_pi]
            pred_text = pred_texts[best_pi]
            gt_norm = _normalize(gt_text)
            pred_norm = _normalize(pred_text)
            cer = compute_cer(gt_norm, pred_norm)
            wer = compute_wer(gt_norm, pred_norm)
            gt_pages = _article_pages(gt_art)
            pred_pages = _article_pages(pred_art)
            page_correct = set(gt_pages) == set(pred_pages)

            matches.append(ArticleMatch(
                gt_index=gi,
                gt_headline=gt_art.get("headline", ""),
                pred_index=best_pi,
                pred_headline=pred_art.get("headline", ""),
                cer=cer,
                wer=wer,
                text_overlap=best_overlap,
                page_span_correct=page_correct,
                gt_pages=gt_pages,
                pred_pages=pred_pages,
            ))
        else:
            matches.append(ArticleMatch(
                gt_index=gi,
                gt_headline=gt_art.get("headline", ""),
                pred_index=None,
                pred_headline=None,
                cer=1.0,
                wer=1.0,
                text_overlap=0.0,
                page_span_correct=False,
                gt_pages=_article_pages(gt_art),
                pred_pages=[],
            ))

    return matches


def evaluate_issue(
    gt_issue: dict[str, tp.Any],
    pred_issue: dict[str, tp.Any],
    overlap_threshold: float = 0.15,
) -> EvalResult:
    gt_articles = _extract_articles(gt_issue)
    pred_articles = _extract_articles(pred_issue)

    matches = match_articles(gt_articles, pred_articles, overlap_threshold)

    matched_gt = sum(1 for m in matches if m.pred_index is not None)
    matched_pred = len({m.pred_index for m in matches if m.pred_index is not None})

    precision = matched_pred / len(pred_articles) if pred_articles else 0.0
    recall = matched_gt / len(gt_articles) if gt_articles else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    matched_cers = [m.cer for m in matches if m.pred_index is not None]
    matched_wers = [m.wer for m in matches if m.pred_index is not None]
    mean_cer = sum(matched_cers) / len(matched_cers) if matched_cers else 1.0
    mean_wer = sum(matched_wers) / len(matched_wers) if matched_wers else 1.0

    page_correct = sum(1 for m in matches if m.page_span_correct)
    page_accuracy = page_correct / len(matches) if matches else 0.0

    return EvalResult(
        matches=matches,
        article_precision=precision,
        article_recall=recall,
        article_f1=f1,
        mean_cer=mean_cer,
        mean_wer=mean_wer,
        page_accuracy=page_accuracy,
        total_gt_articles=len(gt_articles),
        total_pred_articles=len(pred_articles),
    )
