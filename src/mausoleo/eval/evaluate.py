from __future__ import annotations

import dataclasses as dc
import json
import pathlib as pl
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
    mean_cer: float
    mean_wer: float
    weighted_cer: float
    headline_cer: float
    full_text_cer: float
    full_text_wer: float
    page_accuracy: float
    ordering_score: float
    composite_score: float
    total_gt_articles: int
    total_pred_articles: int


def compute_ordering_score(matches: list[ArticleMatch]) -> float:
    paired = [(m.gt_index, m.pred_index) for m in matches if m.pred_index is not None]
    n = len(paired)
    if n < 2:
        return 1.0

    gt_ranks = sorted(range(n), key=lambda i: paired[i][0])
    pred_order = [paired[gt_ranks[i]][1] for i in range(n)]
    rank_map: dict[int, int] = {}
    for rank, val in enumerate(sorted(pred_order)):
        rank_map[val] = rank
    pred_ranks = [rank_map[v] for v in pred_order]

    sum_d_sq = sum((i - pred_ranks[i]) ** 2 for i in range(n))
    max_d_sq = n * (n * n - 1) / 3.0
    return max(0.0, 1.0 - sum_d_sq / max_d_sq)


def match_articles(
    gt_articles: list[dict[str, tp.Any]],
    pred_articles: list[dict[str, tp.Any]],
    overlap_threshold: float = 0.15,
) -> list[ArticleMatch]:
    gt_texts = [article_text(a) for a in gt_articles]
    pred_texts = [article_text(a) for a in pred_articles]

    used_pred: set[int] = set()
    matches: list[ArticleMatch] = []

    for gi, gt_art in enumerate(gt_articles):
        gt_t = gt_texts[gi]
        gt_chars = len(gt_t.strip())
        if gt_chars < 20:
            matches.append(ArticleMatch(
                gt_index=gi, gt_headline=gt_art.get("headline", ""), gt_chars=gt_chars,
                pred_index=None, pred_headline=None,
                cer=1.0, wer=1.0, headline_cer=1.0, text_overlap=0.0,
                page_span_correct=False, gt_pages=article_pages(gt_art), pred_pages=[],
            ))
            continue

        best_pi, best_ov = -1, 0.0
        for pi, pred_t in enumerate(pred_texts):
            if pi in used_pred:
                continue
            ov = text_overlap(gt_t, pred_t)
            if ov > best_ov:
                best_ov = ov
                best_pi = pi

        if best_pi >= 0 and best_ov >= overlap_threshold:
            used_pred.add(best_pi)
            pred_art = pred_articles[best_pi]
            gt_norm = normalize_text(gt_t)
            pred_norm = normalize_text(pred_texts[best_pi])

            gt_h = normalize_text(gt_art.get("headline", "").split("\n")[0])
            pred_h = normalize_text(pred_art.get("headline", "").split("\n")[0] if pred_art.get("headline") else "")
            h_cer = compute_cer(gt_h, pred_h) if gt_h else 0.0

            matches.append(ArticleMatch(
                gt_index=gi, gt_headline=gt_art.get("headline", ""), gt_chars=gt_chars,
                pred_index=best_pi, pred_headline=pred_art.get("headline", ""),
                cer=compute_cer(gt_norm, pred_norm),
                wer=compute_wer(gt_norm, pred_norm),
                headline_cer=h_cer,
                text_overlap=best_ov,
                page_span_correct=set(article_pages(gt_art)) == set(article_pages(pred_art)),
                gt_pages=article_pages(gt_art), pred_pages=article_pages(pred_art),
            ))
        else:
            matches.append(ArticleMatch(
                gt_index=gi, gt_headline=gt_art.get("headline", ""), gt_chars=gt_chars,
                pred_index=None, pred_headline=None,
                cer=1.0, wer=1.0, headline_cer=1.0, text_overlap=0.0,
                page_span_correct=False, gt_pages=article_pages(gt_art), pred_pages=[],
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

    matches = match_articles(gt_articles, pred_articles)

    matched_gt = sum(1 for m in matches if m.pred_index is not None)
    matched_pred = len({m.pred_index for m in matches if m.pred_index is not None})

    precision = matched_pred / len(pred_articles) if pred_articles else 0.0
    recall = matched_gt / len(gt_articles) if gt_articles else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    matched = [m for m in matches if m.pred_index is not None]
    mean_cer = sum(m.cer for m in matched) / len(matched) if matched else 1.0
    mean_wer = sum(m.wer for m in matched) / len(matched) if matched else 1.0

    total_chars = sum(m.gt_chars for m in matched)
    weighted_cer = sum(m.cer * m.gt_chars for m in matched) / total_chars if total_chars > 0 else 1.0

    headline_cers = [m.headline_cer for m in matched if m.gt_headline]
    mean_headline_cer = sum(headline_cers) / len(headline_cers) if headline_cers else 1.0

    gt_full = " ".join(article_text(a) for a in gt_articles)
    pred_full = " ".join(article_text(a) for a in pred_articles)
    full_cer = compute_cer(normalize_text(gt_full), normalize_text(pred_full)) if gt_full.strip() else 1.0
    full_wer = compute_wer(normalize_text(gt_full), normalize_text(pred_full)) if gt_full.strip() else 1.0

    page_correct = sum(1 for m in matches if m.page_span_correct)
    page_accuracy = page_correct / len(matches) if matches else 0.0

    ordering = compute_ordering_score(matches)

    composite = (
        0.40 * (1.0 - min(weighted_cer, 1.0))
        + 0.25 * recall
        + 0.15 * ordering
        + 0.10 * (1.0 - min(mean_headline_cer, 1.0))
        + 0.10 * page_accuracy
    )

    return IssueResult(
        config=config, date=date, matches=matches,
        article_precision=precision, article_recall=recall, article_f1=f1,
        mean_cer=mean_cer, mean_wer=mean_wer,
        weighted_cer=weighted_cer, headline_cer=mean_headline_cer,
        full_text_cer=full_cer, full_text_wer=full_wer,
        page_accuracy=page_accuracy, ordering_score=ordering,
        composite_score=composite,
        total_gt_articles=len(gt_articles), total_pred_articles=len(pred_articles),
    )


def evaluate_all(
    gt_dir: pl.Path = pl.Path("eval/ground_truth"),
    pred_dir: pl.Path = pl.Path("eval/predictions"),
    dates: list[str] | None = None,
) -> list[IssueResult]:
    all_dates = ["1885-06-15", "1910-06-15", "1940-04-01"]
    dates = dates or [d for d in all_dates if (gt_dir / d / "ground_truth.json").exists()]
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
    header = f"{'Config':<45} {'Date':>10} {'CER':>6} {'wCER':>6} {'hCER':>6} {'Rec':>5} {'F1':>5} {'Ord':>5} {'Pg':>5} {'Comp':>5}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.config:<45} {r.date:>10}"
            f" {r.mean_cer:>6.3f}"
            f" {r.weighted_cer:>6.3f}"
            f" {r.headline_cer:>6.3f}"
            f" {r.article_recall:>5.1%}"
            f" {r.article_f1:>5.1%}"
            f" {r.ordering_score:>5.1%}"
            f" {r.page_accuracy:>5.1%}"
            f" {r.composite_score:>5.3f}"
        )
