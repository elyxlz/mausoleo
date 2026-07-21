from __future__ import annotations

import json
import pathlib as pl
import sys

sys.path.insert(0, "src")

from research import EVAL_DATES, evaluate_config

LOG = pl.Path("eval/autoresearch/mausoleobench_log.jsonl")
BUDGET_CAP = 250.0

REFERENCES: dict[str, tuple[str, float]] = {
    "ensemble_30min": ("Multi-source recall oracle ensemble (GT-building reference, far over budget)", 600.0),
    "ensemble_prune5": ("Pruned oracle ensemble (reference, over budget)", 400.0),
}


def main() -> None:
    rows: list[dict[str, object]] = []
    for config, (desc, cost) in REFERENCES.items():
        results = evaluate_config(config, EVAL_DATES)
        if len(results) != len(EVAL_DATES):
            continue
        score = sum(r.mausoleobench_score for r in results.values()) / len(results)
        rows.append({"config": config, "n": 0, "score": round(score, 4), "description": desc,
                     "name": config.replace("ensemble_", ""), "reference": True,
                     "gpu_s_per_page": cost, "budget_ok": cost <= BUDGET_CAP})
    LOG.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"seeded {len(rows)} references -> {LOG} (attempts are appended by the loop)")


if __name__ == "__main__":
    main()
