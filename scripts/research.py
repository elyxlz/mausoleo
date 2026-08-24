from __future__ import annotations

import argparse
import json

from mausoleo.eval.probes import build_lexicon, probe_issue
from mausoleo.eval.scoring import audit_report, evaluate_config, holdout_rows, leaderboard_rows
from mausoleo.paths import EVAL_DATES, PRED_DIR


def cmd_eval(args: argparse.Namespace) -> None:
    results = evaluate_config(args.config, args.dates or list(EVAL_DATES))
    for line in audit_report(args.config, results, None, 0):
        print(line)
    rows = holdout_rows(args.config)
    if rows:
        print(rows[-1])


def cmd_board(args: argparse.Namespace) -> None:
    for line in leaderboard_rows(args.top):
        print(line)


def cmd_holdout(args: argparse.Namespace) -> None:
    for config in args.configs:
        for line in holdout_rows(config):
            print(line)


def cmd_probe(args: argparse.Namespace) -> None:
    lexicon = build_lexicon()
    print(f"lexicon size: {len(lexicon)}")
    for stem in args.stems:
        path = PRED_DIR / f"{stem}.json"
        if not path.exists():
            print(f"{stem}: missing {path}")
            continue
        rendered = " ".join(f"{k}={v:.4g}" for k, v in probe_issue(json.loads(path.read_text()), lexicon).items())
        print(f"{stem}: {rendered}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="research")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_eval = sub.add_parser("eval", help="evaluate predictions: composite + audit + holdout summary")
    p_eval.add_argument("config")
    p_eval.add_argument("--dates", nargs="*", default=None)
    p_eval.set_defaults(fn=cmd_eval)

    p_board = sub.add_parser("board", help="leaderboard over local predictions")
    p_board.add_argument("--top", type=int, default=20)
    p_board.set_defaults(fn=cmd_board)

    p_holdout = sub.add_parser("holdout", help="even/odd article holdout split")
    p_holdout.add_argument("configs", nargs="+")
    p_holdout.set_defaults(fn=cmd_holdout)

    p_probe = sub.add_parser("probe", help="GT-free probe metrics for <config>_<date> stems")
    p_probe.add_argument("stems", nargs="+")
    p_probe.set_defaults(fn=cmd_probe)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
