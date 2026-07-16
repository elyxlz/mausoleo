from __future__ import annotations

import argparse
import json
import pathlib as pl
import subprocess
import sys
import time
import typing as tp

sys.path.insert(0, "src")

from mausoleo.eval.evaluate import IssueResult, evaluate_issue

REMOTE = "audiogen@81.105.49.222"
SSH_PORT = "62022"
REMOTE_DIR = "mausoleo_di_roma"
GT_DIR = pl.Path("eval/ground_truth")
PRED_DIR = pl.Path("eval/predictions")
RUNS_LOG = pl.Path("eval/autoresearch/runs.jsonl")
EVAL_DATES = ("1885-06-15", "1910-06-15")
BUDGET_GPU_S_PER_PAGE = (6.9, 13.9)

RSYNC_EXCLUDES = ["--exclude=.venv", "--exclude=.git", "--exclude=__pycache__", "--exclude=eval/predictions", "--exclude=eval/ground_truth"]


def sync_repo() -> None:
    cmd = ["rsync", "-az", *RSYNC_EXCLUDES, "-e", f"ssh -p {SSH_PORT}", "./", f"{REMOTE}:~/{REMOTE_DIR}/"]
    subprocess.run(cmd, check=True)


def run_remote(config: str, dates: tp.Sequence[str], force: bool) -> float:
    force_flag = " --force" if force else ""
    date_args = " ".join(dates)
    remote_cmd = (
        f"cd {REMOTE_DIR} && find src/ -name __pycache__ -exec rm -rf {{}} + 2>/dev/null; "
        f".venv/bin/python scripts/run_real_ocr.py {config} {date_args}{force_flag}"
    )
    start = time.monotonic()
    subprocess.run(["ssh", "-p", SSH_PORT, REMOTE, remote_cmd], check=True)
    return time.monotonic() - start


def fetch_predictions(config: str, dates: tp.Sequence[str]) -> None:
    for date in dates:
        remote_path = f"{REMOTE}:~/{REMOTE_DIR}/eval/predictions/{config}_{date}.json"
        subprocess.run(["scp", "-P", SSH_PORT, "-q", remote_path, str(PRED_DIR)], check=True)


def count_pages(dates: tp.Sequence[str]) -> int:
    return sum(len(list((GT_DIR / date).glob("*.jpeg"))) for date in dates)


def evaluate_config(config: str, dates: tp.Sequence[str]) -> dict[str, IssueResult]:
    results: dict[str, IssueResult] = {}
    for date in dates:
        gt_path = GT_DIR / date / "ground_truth.json"
        pred_path = PRED_DIR / f"{config}_{date}.json"
        if not gt_path.exists() or not pred_path.exists():
            continue
        gt_issue = json.loads(gt_path.read_text())
        pred_issue = json.loads(pred_path.read_text())
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


def holdout_report(config: str) -> list[str]:
    proc = subprocess.run(
        [sys.executable, "scripts/eval_holdout.py", config],
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip().splitlines()[-1:] if proc.returncode == 0 else ["holdout: failed"]


def record_run(config: str, dates: tp.Sequence[str], wall_s: float, results: dict[str, IssueResult]) -> None:
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M"),
        "config": config,
        "dates": list(dates),
        "wall_s": round(wall_s, 1),
        "pages": count_pages(dates),
        "composites": {d: round(r.composite_score, 4) for d, r in results.items()},
    }
    with RUNS_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def cmd_run(args: argparse.Namespace) -> None:
    dates = args.dates or list(EVAL_DATES)
    print("[research] syncing to ripperred...", flush=True)
    sync_repo()
    print(f"[research] running {args.config} on {dates}...", flush=True)
    wall_s = run_remote(args.config, dates, args.force)
    fetch_predictions(args.config, dates)
    results = evaluate_config(args.config, dates)
    record_run(args.config, dates, wall_s, results)
    for line in audit_report(args.config, results, wall_s, count_pages(dates)):
        print(line)
    for line in holdout_report(args.config):
        print(line)
    print("[research] remember: inspect concrete predictions, log to log.jsonl with mechanism line, update registry.md")


def cmd_eval(args: argparse.Namespace) -> None:
    dates = args.dates or list(EVAL_DATES)
    results = evaluate_config(args.config, dates)
    for line in audit_report(args.config, results, None, 0):
        print(line)
    for line in holdout_report(args.config):
        print(line)


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


def main() -> None:
    parser = argparse.ArgumentParser(prog="research")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="sync, run on ripperred, fetch, evaluate, audit")
    p_run.add_argument("config")
    p_run.add_argument("--dates", nargs="*", default=None)
    p_run.add_argument("--force", action="store_true")
    p_run.set_defaults(fn=cmd_run)

    p_eval = sub.add_parser("eval", help="evaluate existing local predictions")
    p_eval.add_argument("config")
    p_eval.add_argument("--dates", nargs="*", default=None)
    p_eval.set_defaults(fn=cmd_eval)

    p_board = sub.add_parser("board", help="leaderboard over local predictions")
    p_board.add_argument("--top", type=int, default=25)
    p_board.set_defaults(fn=cmd_board)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
