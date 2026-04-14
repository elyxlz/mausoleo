from __future__ import annotations

import dataclasses as dc
import json
import pathlib as pl
import re
import typing as tp


GT_DIR = pl.Path("eval/ground_truth")
PRED_DIR = pl.Path("eval/predictions")
DATES = ["1885-06-15", "1910-06-15", "1940-04-01"]


def _compute_cer(reference: str, hypothesis: str) -> float:
    import jiwer

    if not reference:
        return 0.0 if not hypothesis else 1.0
    return jiwer.cer(reference, hypothesis)  # type: ignore[no-any-return]


def _compute_wer(reference: str, hypothesis: str) -> float:
    import jiwer

    if not reference:
        return 0.0 if not hypothesis else 1.0
    return jiwer.wer(reference, hypothesis)  # type: ignore[no-any-return]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _text_overlap(a: str, b: str) -> float:
    a_words = set(_normalize(a).split())
    b_words = set(_normalize(b).split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words | b_words)


def _article_text(article: dict[str, tp.Any]) -> str:
    return "\n".join(p.get("text", "") for p in article.get("paragraphs", []))


def _article_pages(article: dict[str, tp.Any]) -> list[int]:
    return article.get("page_span", [])


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
class IssueResult:
    config: str
    date: str
    matches: list[ArticleMatch]
    article_precision: float
    article_recall: float
    article_f1: float
    mean_cer: float
    mean_wer: float
    full_text_cer: float
    full_text_wer: float
    page_accuracy: float
    total_gt_articles: int
    total_pred_articles: int


def _match_articles(
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
                gt_index=gi, gt_headline=gt_art.get("headline", ""),
                pred_index=None, pred_headline=None,
                cer=1.0, wer=1.0, text_overlap=0.0,
                page_span_correct=False, gt_pages=_article_pages(gt_art), pred_pages=[],
            ))
            continue

        best_pi, best_overlap = -1, 0.0
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
            gt_norm = _normalize(gt_text)
            pred_norm = _normalize(pred_texts[best_pi])
            matches.append(ArticleMatch(
                gt_index=gi, gt_headline=gt_art.get("headline", ""),
                pred_index=best_pi, pred_headline=pred_art.get("headline", ""),
                cer=_compute_cer(gt_norm, pred_norm),
                wer=_compute_wer(gt_norm, pred_norm),
                text_overlap=best_overlap,
                page_span_correct=set(_article_pages(gt_art)) == set(_article_pages(pred_art)),
                gt_pages=_article_pages(gt_art), pred_pages=_article_pages(pred_art),
            ))
        else:
            matches.append(ArticleMatch(
                gt_index=gi, gt_headline=gt_art.get("headline", ""),
                pred_index=None, pred_headline=None,
                cer=1.0, wer=1.0, text_overlap=0.0,
                page_span_correct=False, gt_pages=_article_pages(gt_art), pred_pages=[],
            ))

    return matches


def evaluate_issue(
    gt_issue: dict[str, tp.Any],
    pred_issue: dict[str, tp.Any],
    config: str = "",
    date: str = "",
) -> IssueResult:
    gt_articles = gt_issue.get("articles", [])
    pred_articles = pred_issue.get("articles", [])

    matches = _match_articles(gt_articles, pred_articles)

    matched_gt = sum(1 for m in matches if m.pred_index is not None)
    matched_pred = len({m.pred_index for m in matches if m.pred_index is not None})

    precision = matched_pred / len(pred_articles) if pred_articles else 0.0
    recall = matched_gt / len(gt_articles) if gt_articles else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    matched_cers = [m.cer for m in matches if m.pred_index is not None]
    matched_wers = [m.wer for m in matches if m.pred_index is not None]
    mean_cer = sum(matched_cers) / len(matched_cers) if matched_cers else 1.0
    mean_wer = sum(matched_wers) / len(matched_wers) if matched_wers else 1.0

    gt_full = " ".join(_article_text(a) for a in gt_articles)
    pred_full = " ".join(_article_text(a) for a in pred_articles)
    full_cer = _compute_cer(_normalize(gt_full), _normalize(pred_full)) if gt_full.strip() else 1.0
    full_wer = _compute_wer(_normalize(gt_full), _normalize(pred_full)) if gt_full.strip() else 1.0

    page_correct = sum(1 for m in matches if m.page_span_correct)
    page_accuracy = page_correct / len(matches) if matches else 0.0

    return IssueResult(
        config=config, date=date, matches=matches,
        article_precision=precision, article_recall=recall, article_f1=f1,
        mean_cer=mean_cer, mean_wer=mean_wer,
        full_text_cer=full_cer, full_text_wer=full_wer,
        page_accuracy=page_accuracy,
        total_gt_articles=len(gt_articles), total_pred_articles=len(pred_articles),
    )


def evaluate_all(
    gt_dir: pl.Path = GT_DIR,
    pred_dir: pl.Path = PRED_DIR,
    dates: list[str] | None = None,
) -> list[IssueResult]:
    dates = dates or [d for d in DATES if (gt_dir / d / "ground_truth.json").exists()]
    results: list[IssueResult] = []

    for date in dates:
        gt_path = gt_dir / date / "ground_truth.json"
        if not gt_path.exists():
            continue
        gt_issue = json.loads(gt_path.read_text())

        for pred_path in sorted(pred_dir.glob(f"*_{date}.json")):
            cfg = pred_path.stem.replace(f"_{date}", "")
            try:
                pred_issue = json.loads(pred_path.read_text())
            except Exception:
                continue
            results.append(evaluate_issue(gt_issue, pred_issue, config=cfg, date=date))

    return results


def print_results(results: list[IssueResult]) -> None:
    results = sorted(results, key=lambda r: r.mean_cer)
    header = f"{'Config':<45} {'Date':>10} {'ArtCER':>7} {'ArtWER':>7} {'FullCER':>7} {'Recall':>6} {'F1':>6} {'Pages':>6} {'GT':>3} {'Pred':>4}"
    print(header)
    print("-" * len(header))
    for r in results:
        ac = f"{r.mean_cer:.3f}" if r.mean_cer < 10 else f"{r.mean_cer:.1f}"
        aw = f"{r.mean_wer:.3f}" if r.mean_wer < 10 else f"{r.mean_wer:.1f}"
        fc = f"{r.full_text_cer:.3f}" if r.full_text_cer < 10 else f"{r.full_text_cer:.1f}"
        print(f"{r.config:<45} {r.date:>10} {ac:>7} {aw:>7} {fc:>7} {r.article_recall:>6.1%} {r.article_f1:>6.1%} {r.page_accuracy:>6.1%} {r.total_gt_articles:>3} {r.total_pred_articles:>4}")
