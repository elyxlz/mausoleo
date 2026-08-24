from __future__ import annotations

import json
import typing as tp

from mausoleo.eval.evaluate import IssueResult, evaluate_issue
from mausoleo.paths import BUDGET_CAP, EVAL_DATES, GT_DIR, PRED_DIR


def evaluate_config(config: str, dates: tp.Sequence[str]) -> dict[str, IssueResult]:
    results: dict[str, IssueResult] = {}
    for date in dates:
        gt_path = GT_DIR / date / "ground_truth.json"
        pred_path = PRED_DIR / f"{config}_{date}.json"
        if not gt_path.exists() or not pred_path.exists():
            continue
        try:
            gt_issue = json.loads(gt_path.read_text())
            pred_issue = json.loads(pred_path.read_text())
        except json.JSONDecodeError:
            continue
        results[date] = evaluate_issue(gt_issue, pred_issue, config=config, date=date)
    return results


def audit_report(config: str, results: dict[str, IssueResult], wall_s: float | None, pages: int) -> list[str]:
    lines: list[str] = []
    if len(results) == len(EVAL_DATES):
        avg = sum(r.mausoleobench_score for r in results.values()) / len(results)
        lines.append(f"avg MausoleoBench: {avg:.4f}")
    for date, r in results.items():
        lines.append(
            f"{date}: mbench={r.mausoleobench_score:.4f} wCER={r.weighted_cer:.3f} cer={r.mean_cer:.3f} gF1={r.article_gated_f1:.3f} recall={r.article_recall:.3f}"
            f" hCER={r.headline_cer:.3f} ord={r.ordering_score:.3f} pgacc={r.page_accuracy:.3f}"
            f" preds={r.total_pred_articles}/gt={r.total_gt_articles}"
        )
        blob_matches = [m for m in r.matches if m.pred_index is not None and m.cer > 2.0]
        if blob_matches:
            worst = max(blob_matches, key=lambda m: m.cer)
            lines.append(f"  AUDIT giant-blob matches: {len(blob_matches)} (worst CER {worst.cer:.1f}: {worst.gt_headline[:40]!r})")
        if r.total_pred_articles > 6 * max(r.total_gt_articles, 1):
            lines.append(f"  AUDIT overgeneration: {r.total_pred_articles} preds vs {r.total_gt_articles} GT")
    if wall_s is not None and pages:
        sec_per_page = wall_s / pages
        verdict = "within budget" if sec_per_page <= BUDGET_CAP else "OVER BUDGET (research-only)"
        lines.append(f"timing: {wall_s:.0f}s wall for {pages} pages = {sec_per_page:.1f} sec/page (cold) -> {verdict}")
    return lines


def split_gt(gt_issue: dict[str, tp.Any]) -> tuple[dict[str, tp.Any], dict[str, tp.Any]]:
    articles = gt_issue.get("articles", [])
    return {**gt_issue, "articles": articles[0::2]}, {**gt_issue, "articles": articles[1::2]}


def holdout_rows(config: str) -> list[str]:
    lines: list[str] = []
    tune_scores, holdout_scores = [], []
    for date in EVAL_DATES:
        gt_path = GT_DIR / date / "ground_truth.json"
        pred_path = PRED_DIR / f"{config}_{date}.json"
        if not gt_path.exists() or not pred_path.exists():
            continue
        gt_issue = json.loads(gt_path.read_text())
        pred_issue = json.loads(pred_path.read_text())
        tune, holdout = split_gt(gt_issue)
        t = evaluate_issue(tune, pred_issue, config=config, date=date).mausoleobench_score
        h = evaluate_issue(holdout, pred_issue, config=config, date=date).mausoleobench_score
        tune_scores.append(t)
        holdout_scores.append(h)
        lines.append(f"{config:<45} {date:>10} tune={t:.4f} holdout={h:.4f} gap={h - t:+.4f}")
    if tune_scores:
        t, h = sum(tune_scores) / len(tune_scores), sum(holdout_scores) / len(holdout_scores)
        lines.append(f"{config:<45} {'AVG':>10} tune={t:.4f} holdout={h:.4f} gap={h - t:+.4f}")
    return lines


def leaderboard_rows(top: int = 20) -> list[str]:
    configs = sorted({p.stem.replace(f"_{d}", "") for d in EVAL_DATES for p in PRED_DIR.glob(f"*_{d}.json")})
    full: list[tuple[float, str, str]] = []
    partial: list[tuple[int, str]] = []
    for config in configs:
        results = evaluate_config(config, EVAL_DATES)
        if not results:
            continue
        detail = " ".join(f"{d[:4]}={results[d].mausoleobench_score:.3f}" if d in results else f"{d[:4]}=--" for d in EVAL_DATES)
        if len(results) == len(EVAL_DATES):
            full.append((sum(r.mausoleobench_score for r in results.values()) / len(EVAL_DATES), config, detail))
        else:
            partial.append((len(results), f"{config:<45} [{len(results)}/{len(EVAL_DATES)}]  {detail}"))
    lines = [f"{avg:.4f}  {config:<45} {detail}" for avg, config, detail in sorted(full, reverse=True)[:top]]
    if partial:
        lines.append("\n-- partial coverage (not ranked) --")
        lines.extend(f"        {line}" for _, line in sorted(partial, reverse=True))
    return lines
