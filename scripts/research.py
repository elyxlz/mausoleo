from __future__ import annotations

import argparse
import collections
import json
import pathlib as pl
import re
import sys
import typing as tp

sys.path.insert(0, "src")

from mausoleo.eval.evaluate import IssueResult, evaluate_issue

GT_DIR = pl.Path("eval/ground_truth")
PRED_DIR = pl.Path("eval/predictions")
EVAL_DATES = ("1885-06-15", "1910-06-15")
BUDGET_GPU_S_PER_PAGE = (6.9, 13.9)
LEXICON_DATES = ("1885-06-15", "1910-06-15")
WORD_RE = re.compile(r"[a-zàèéìòóù]{2,}")


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
        avg = sum(r.composite_score for r in results.values()) / len(results)
        lines.append(f"avg composite: {avg:.4f}")
    for date, r in results.items():
        lines.append(
            f"{date}: comp={r.composite_score:.4f} wCER={r.weighted_cer:.3f} recall={r.article_recall:.3f}"
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
        gpu_s_page = wall_s / pages
        low, high = BUDGET_GPU_S_PER_PAGE
        verdict = "WITHIN 1-week budget" if gpu_s_page <= low else "within 2-week cap" if gpu_s_page <= high else "OVER BUDGET (research-only)"
        lines.append(f"timing: {wall_s:.0f}s wall for {pages} pages = {gpu_s_page:.1f} s/page (cold, per-issue) -> {verdict}")
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
        t = evaluate_issue(tune, pred_issue, config=config, date=date).composite_score
        h = evaluate_issue(holdout, pred_issue, config=config, date=date).composite_score
        tune_scores.append(t)
        holdout_scores.append(h)
        lines.append(f"{config:<45} {date:>10} tune={t:.4f} holdout={h:.4f} gap={h - t:+.4f}")
    if tune_scores:
        t, h = sum(tune_scores) / len(tune_scores), sum(holdout_scores) / len(holdout_scores)
        lines.append(f"{config:<45} {'AVG':>10} tune={t:.4f} holdout={h:.4f} gap={h - t:+.4f}")
    return lines


def build_lexicon() -> frozenset[str]:
    words: set[str] = set()
    for date in LEXICON_DATES:
        gt_path = GT_DIR / date / "ground_truth.json"
        if not gt_path.exists():
            continue
        for article in json.loads(gt_path.read_text()).get("articles", []):
            for paragraph in article.get("paragraphs", []):
                words.update(WORD_RE.findall(paragraph.get("text", "").lower()))
    return frozenset(words)


def article_text(article: dict[str, tp.Any]) -> str:
    return "\n".join(p.get("text", "") for p in article.get("paragraphs", []))


def lexicon_validity(text: str, lexicon: frozenset[str]) -> float:
    tokens = WORD_RE.findall(text.lower())
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in lexicon) / len(tokens)


def repetition_rate(text: str) -> float:
    tokens = text.lower().split()
    if len(tokens) < 20:
        return 0.0
    trigrams = [" ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    most_common = collections.Counter(trigrams).most_common(1)[0][1]
    return most_common / len(trigrams)


def probe_issue(pred_issue: dict[str, tp.Any], lexicon: frozenset[str]) -> dict[str, float]:
    articles = pred_issue.get("articles", [])
    texts = [article_text(a) for a in articles]
    lengths = sorted(len(t) for t in texts)
    n = len(texts)
    if n == 0:
        return {"articles": 0.0}
    full_text = " ".join(texts)
    return {
        "articles": float(n),
        "lexicon_validity": lexicon_validity(full_text, lexicon),
        "mean_repetition": sum(repetition_rate(t) for t in texts) / n,
        "median_chars": float(lengths[n // 2]),
        "total_chars": float(sum(lengths)),
        "share_tiny_lt50": sum(1 for length in lengths if length < 50) / n,
        "share_huge_gt5k": sum(1 for length in lengths if length > 5000) / n,
        "share_high_repetition": sum(1 for t in texts if repetition_rate(t) > 0.10) / n,
    }


def cmd_eval(args: argparse.Namespace) -> None:
    dates = args.dates or list(EVAL_DATES)
    results = evaluate_config(args.config, dates)
    for line in audit_report(args.config, results, None, 0):
        print(line)
    rows = holdout_rows(args.config)
    if rows:
        print(rows[-1])


def cmd_board(args: argparse.Namespace) -> None:
    configs = sorted({p.stem.replace(f"_{d}", "") for d in EVAL_DATES for p in PRED_DIR.glob(f"*_{d}.json")})
    rows: list[tuple[float, str, str]] = []
    for config in configs:
        results = evaluate_config(config, EVAL_DATES)
        if not results:
            continue
        avg = sum(r.composite_score for r in results.values()) / len(results)
        detail = " ".join(f"{d[:4]}={r.composite_score:.3f}" for d, r in results.items())
        rows.append((avg, config, detail))
    for avg, config, detail in sorted(rows, reverse=True)[: args.top]:
        print(f"{avg:.4f}  {config:<45} {detail}")


def cmd_holdout(args: argparse.Namespace) -> None:
    for config in args.configs:
        for line in holdout_rows(config):
            print(line)


def cmd_probe(args: argparse.Namespace) -> None:
    lexicon = build_lexicon()
    print(f"lexicon size: {len(lexicon)}")
    for stem in args.stems:
        pred_path = PRED_DIR / f"{stem}.json"
        if not pred_path.exists():
            print(f"{stem}: missing {pred_path}")
            continue
        metrics = probe_issue(json.loads(pred_path.read_text()), lexicon)
        rendered = " ".join(f"{k}={v:.4g}" for k, v in metrics.items())
        print(f"{stem}: {rendered}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="research")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_eval = sub.add_parser("eval", help="evaluate predictions: composite + audit + holdout summary")
    p_eval.add_argument("config")
    p_eval.add_argument("--dates", nargs="*", default=None)
    p_eval.set_defaults(fn=cmd_eval)

    p_board = sub.add_parser("board", help="leaderboard over local predictions")
    p_board.add_argument("--top", type=int, default=25)
    p_board.set_defaults(fn=cmd_board)

    p_holdout = sub.add_parser("holdout", help="tune/holdout half scores per config")
    p_holdout.add_argument("configs", nargs="+")
    p_holdout.set_defaults(fn=cmd_holdout)

    p_probe = sub.add_parser("probe", help="GT-free probe metrics for <config>_<date> stems")
    p_probe.add_argument("stems", nargs="+")
    p_probe.set_defaults(fn=cmd_probe)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
