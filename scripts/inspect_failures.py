from __future__ import annotations

import json
import sys

from mausoleo.eval.evaluate import article_text, evaluate_issue


def inspect(config: str, date: str) -> None:
    gt = json.loads(open(f"eval/ground_truth/{date}/ground_truth.json").read())
    pred = json.loads(open(f"eval/predictions/{config}_{date}.json").read())
    r = evaluate_issue(gt, pred, config=config, date=date)

    print(f"Config: {config} | Date: {date}")
    print(f"Composite: {r.composite_score:.3f} | CER: {r.mean_cer:.3f} | wCER: {r.weighted_cer:.3f} | Recall: {r.article_recall:.1%} | Pred articles: {r.total_pred_articles}")
    print()

    print("=== WORST 15 MATCHED ARTICLES (highest CER) ===")
    matched = sorted([m for m in r.matches if m.pred_index is not None], key=lambda m: -m.cer)
    for m in matched[:15]:
        gt_t = article_text(gt["articles"][m.gt_index])[:100]
        pred_t = article_text(pred["articles"][m.pred_index])[:100]
        hl = (m.gt_headline or "none")[:60]
        print(f"  GT[{m.gt_index}] '{hl}' CER={m.cer:.3f} chars={m.gt_chars} pg={m.gt_pages}")
        print(f"    GT:   {gt_t}")
        print(f"    PRED: {pred_t}")
        print()

    print("=== UNMATCHED GT ARTICLES (missed entirely) ===")
    unmatched = [m for m in r.matches if m.pred_index is None]
    for m in unmatched[:25]:
        gt_t = article_text(gt["articles"][m.gt_index])[:120]
        hl = (m.gt_headline or "none")[:60]
        print(f"  GT[{m.gt_index}] '{hl}' chars={m.gt_chars} pg={m.gt_pages}")
        print(f"    {gt_t}")
        print()

    print(f"Total unmatched: {len(unmatched)} / {r.total_gt_articles}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "col3_qwen3_8b_v2_structured"
    date = sys.argv[2] if len(sys.argv) > 2 else "1910-06-15"
    inspect(config, date)
