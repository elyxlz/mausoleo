from __future__ import annotations

import json
import pathlib as pl
import sys
import typing as tp

sys.path.insert(0, "src")

from mausoleo.eval.evaluate import evaluate_issue

GT_DIR = pl.Path("eval/ground_truth")
PRED_DIR = pl.Path("eval/predictions")
DATES = ("1885-06-15", "1910-06-15")


def split_gt(gt_issue: dict[str, tp.Any]) -> tuple[dict[str, tp.Any], dict[str, tp.Any]]:
    articles = gt_issue.get("articles", [])
    tune = {**gt_issue, "articles": articles[0::2]}
    holdout = {**gt_issue, "articles": articles[1::2]}
    return tune, holdout


def evaluate_halves(config: str, date: str) -> tuple[float, float] | None:
    gt_path = GT_DIR / date / "ground_truth.json"
    pred_path = PRED_DIR / f"{config}_{date}.json"
    if not gt_path.exists() or not pred_path.exists():
        return None
    gt_issue = json.loads(gt_path.read_text())
    pred_issue = json.loads(pred_path.read_text())
    tune, holdout = split_gt(gt_issue)
    tune_score = evaluate_issue(tune, pred_issue, config=config, date=date).composite_score
    holdout_score = evaluate_issue(holdout, pred_issue, config=config, date=date).composite_score
    return tune_score, holdout_score


def main() -> None:
    configs = sys.argv[1:]
    if not configs:
        print("usage: eval_holdout.py <config> [<config> ...]")
        raise SystemExit(1)
    print(f"{'config':<40} {'date':>10} {'tune':>7} {'holdout':>7} {'gap':>7}")
    for config in configs:
        tune_avg, holdout_avg = [], []
        for date in DATES:
            halves = evaluate_halves(config, date)
            if halves is None:
                print(f"{config:<40} {date:>10} missing prediction or GT")
                continue
            tune_score, holdout_score = halves
            tune_avg.append(tune_score)
            holdout_avg.append(holdout_score)
            print(f"{config:<40} {date:>10} {tune_score:>7.4f} {holdout_score:>7.4f} {holdout_score - tune_score:>+7.4f}")
        if tune_avg and holdout_avg:
            t, h = sum(tune_avg) / len(tune_avg), sum(holdout_avg) / len(holdout_avg)
            print(f"{config:<40} {'AVG':>10} {t:>7.4f} {h:>7.4f} {h - t:>+7.4f}")


if __name__ == "__main__":
    main()
