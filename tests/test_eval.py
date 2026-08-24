from __future__ import annotations

import typing as tp

from mausoleo.eval.evaluate import compute_cer, compute_ordering_score, compute_wer, evaluate_issue, match_articles


def _article(headline: str, text: str, pages: list[int]) -> dict[str, tp.Any]:
    return {"headline": headline, "paragraphs": [{"text": text}], "page_span": pages}


LONG_A = "Il consiglio comunale ha approvato ieri sera il nuovo bilancio della città di Roma dopo lunga discussione."
LONG_B = "Un violento temporale si è abbattuto ieri sul litorale laziale causando gravi danni alle campagne circostanti."


def test_cer_identical() -> None:
    assert compute_cer("hello world", "hello world") == 0.0


def test_cer_different() -> None:
    assert 0.0 < compute_cer("hello", "hallo") < 1.0


def test_wer_identical() -> None:
    assert compute_wer("hello world", "hello world") == 0.0


def test_wer_different() -> None:
    assert 0.0 < compute_wer("hello world", "hallo world") <= 1.0


def test_ordering_degenerate_is_zero() -> None:
    gt = {"articles": [_article("Uno", LONG_A, [1])]}
    matches = match_articles(gt["articles"], gt["articles"])
    assert compute_ordering_score(matches) == 0.0


def test_evaluate_issue_perfect_match() -> None:
    issue = {"articles": [_article("Uno", LONG_A, [1]), _article("Due", LONG_B, [2])]}
    result = evaluate_issue(issue, issue)
    assert result.mean_cer == 0.0
    assert result.weighted_cer == 0.0
    assert result.article_recall == 1.0
    assert result.article_precision == 1.0
    assert result.mausoleobench_score > 0.95


def test_spam_lowers_composite() -> None:
    gt = {"articles": [_article("Uno", LONG_A, [1]), _article("Due", LONG_B, [2])]}
    fabricated = [_article("", f"articolo inventato numero {i} senza alcuna corrispondenza reale", [1]) for i in range(50)]
    spammed = {"articles": [*gt["articles"], *fabricated]}
    assert evaluate_issue(gt, spammed).mausoleobench_score < evaluate_issue(gt, gt).mausoleobench_score - 0.05


def test_unmatched_gt_counts_in_wcer() -> None:
    gt = {"articles": [_article("Uno", LONG_A, [1]), _article("Due", LONG_B, [2])]}
    partial = {"articles": [gt["articles"][0]]}
    result = evaluate_issue(gt, partial)
    assert result.weighted_cer > 0.4
    assert result.mausoleobench_score < 0.6


def test_empty_prediction_scores_near_zero() -> None:
    gt = {"articles": [_article("Uno", LONG_A, [1]), _article("Due", LONG_B, [2])]}
    result = evaluate_issue(gt, {"articles": []})
    assert result.mausoleobench_score < 0.05
