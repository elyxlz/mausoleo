from __future__ import annotations

import json
import pathlib as pl
import sys

from mausoleo.eval.article_metrics import evaluate_issue

GT_DIR = pl.Path("eval/ground_truth")
PRED_DIR = pl.Path("eval/predictions")
DATES = ["1885-06-15", "1910-06-15", "1940-04-01"]


def run_eval(dates: list[str] | None = None) -> None:
    dates = dates or [d for d in DATES if (GT_DIR / d / "ground_truth.json").exists()]

    results: list[tuple[str, str, float, float, float, float, float, int, int]] = []

    for date in dates:
        gt_path = GT_DIR / date / "ground_truth.json"
        if not gt_path.exists():
            print(f"No GT for {date}, skipping")
            continue

        gt_issue = json.loads(gt_path.read_text())
        n_gt = len(gt_issue.get("articles", []))

        for pred_path in sorted(PRED_DIR.glob(f"*_{date}.json")):
            cfg = pred_path.stem.replace(f"_{date}", "")
            try:
                pred_issue = json.loads(pred_path.read_text())
            except Exception:
                continue

            result = evaluate_issue(gt_issue, pred_issue)
            results.append((
                cfg, date, result.mean_cer, result.mean_wer,
                result.article_recall, result.article_f1,
                result.page_accuracy, n_gt, result.total_pred_articles,
            ))

    results.sort(key=lambda x: x[2])

    print(f"{'Config':<50} {'Date':>10} {'CER':>6} {'WER':>6} {'Recall':>6} {'F1':>6} {'Pages':>6} {'GT':>3} {'Pred':>4}")
    print("-" * 105)
    for cfg, date, cer, wer, recall, f1, pages, n_gt, n_pred in results:
        cer_s = f"{cer:.3f}" if cer < 10 else f"{cer:.1f}"
        wer_s = f"{wer:.3f}" if wer < 10 else f"{wer:.1f}"
        print(f"{cfg:<50} {date:>10} {cer_s:>6} {wer_s:>6} {recall:>6.1%} {f1:>6.1%} {pages:>6.1%} {n_gt:>3} {n_pred:>4}")


if __name__ == "__main__":
    dates = sys.argv[1:] if len(sys.argv) > 1 else None
    run_eval(dates)
