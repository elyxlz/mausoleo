from __future__ import annotations

import pathlib as pl

GT_DIR = pl.Path("eval/ground_truth")
PRED_DIR = pl.Path("eval/predictions")
LOG_PATH = pl.Path("eval/autoresearch/mausoleobench_log.jsonl")

EVAL_DATES = ("1885-06-15", "1895-06-15", "1910-06-15", "1925-06-15", "1935-06-15", "1952-06-15")
BUDGET_CAP = 200.0
