from __future__ import annotations

import json
import pathlib as pl
import sys
import time

from mausoleo.eval.evaluate import evaluate_issue

GT_DIR = pl.Path("eval/ground_truth")
PRED_DIR = pl.Path("eval/predictions")
LOG_PATH = pl.Path("eval/autoresearch/log.jsonl")
DATE = "1885-06-15"


def run_and_eval(config_name: str) -> dict:
    gt_path = GT_DIR / DATE / "ground_truth.json"
    pred_path = PRED_DIR / f"{config_name}_{DATE}.json"

    if not pred_path.exists():
        return {"error": f"No prediction at {pred_path}"}

    gt_issue = json.loads(gt_path.read_text())
    pred_issue = json.loads(pred_path.read_text())

    result = evaluate_issue(gt_issue, pred_issue, config=config_name, date=DATE)

    entry = {
        "config": config_name,
        "date": DATE,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "article_cer": round(result.mean_cer, 4),
        "article_wer": round(result.mean_wer, 4),
        "full_text_cer": round(result.full_text_cer, 4),
        "article_recall": round(result.article_recall, 4),
        "article_f1": round(result.article_f1, 4),
        "page_accuracy": round(result.page_accuracy, 4),
        "gt_articles": result.total_gt_articles,
        "pred_articles": result.total_pred_articles,
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: run_experiment.py <config_name>")
        sys.exit(1)
    result = run_and_eval(sys.argv[1])
    print(json.dumps(result, indent=2))
