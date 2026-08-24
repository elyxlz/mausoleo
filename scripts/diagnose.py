from __future__ import annotations

import argparse

from mausoleo.eval.diagnostics import report_lines
from mausoleo.paths import EVAL_DATES


def main() -> None:
    parser = argparse.ArgumentParser(prog="diagnose", description="per-issue prediction health for a config")
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--dates", nargs="*", default=None)
    args = parser.parse_args()
    for config in args.configs:
        print(f"=== {config}")
        for line in report_lines(config, args.dates or EVAL_DATES):
            print(line)
        print()


if __name__ == "__main__":
    main()
