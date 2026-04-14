from __future__ import annotations

import json
import pathlib as pl
import re

from mausoleo.eval.article_metrics import evaluate_issue

GT_DIR = pl.Path("eval/ground_truth")
PRED_DIR = pl.Path("eval/predictions")
DATES = ["1885-06-15"]


def main() -> None:
    results: list[tuple[str, float, float, float, float, float, int, int]] = []

    for date in DATES:
        gt_path = GT_DIR / date / "ground_truth.json"
        if not gt_path.exists():
            print(f"No GT for {date}")
            continue

        gt_issue = json.loads(gt_path.read_text())

        for pred_path in sorted(PRED_DIR.glob(f"*_{date}.json")):
            cfg = pred_path.stem.replace(f"_{date}", "")
            try:
                pred_issue = json.loads(pred_path.read_text())
            except Exception:
                continue

            result = evaluate_issue(gt_issue, pred_issue)

            results.append((
                cfg,
                result.mean_cer,
                result.mean_wer,
                result.article_recall,
                result.article_f1,
                result.page_accuracy,
                result.total_gt_articles,
                result.total_pred_articles,
            ))

    results.sort(key=lambda x: x[1])

    print(f"{'Config':<50} {'CER':>6} {'WER':>6} {'Recall':>6} {'F1':>6} {'Pages':>6} {'GT':>3} {'Pred':>4}")
    print("-" * 95)
    for cfg, cer, wer, recall, f1, pages, n_gt, n_pred in results:
        cer_s = f"{cer:.3f}" if cer < 10 else f"{cer:.1f}"
        wer_s = f"{wer:.3f}" if wer < 10 else f"{wer:.1f}"
        print(f"{cfg:<50} {cer_s:>6} {wer_s:>6} {recall:>6.1%} {f1:>6.1%} {pages:>6.1%} {n_gt:>3} {n_pred:>4}")


if __name__ == "__main__":
    main()
