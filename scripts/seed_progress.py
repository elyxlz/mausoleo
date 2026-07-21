from __future__ import annotations

import json
import pathlib as pl
import re
import sys

sys.path.insert(0, "src")

from research import EVAL_DATES, evaluate_config

LOG = pl.Path("eval/autoresearch/mausoleobench_log.jsonl")
BUDGET_CAP = 13.9

GPU_COST: dict[str, float] = {
    "exp_045_qwen3vl_vllm": 136.0, "exp_055_col6_ads_prompt": 180.0,
    "exp_097_col4_qwen3vl_vllm": 150.0, "exp_102_fullpage_vllm": 74.0,
    "exp_107_fullpage_qwen25vl": 74.0, "exp_138_col4_qwen25_vllm": 150.0,
    "exp_140_yolo_smallregion_vllm": 90.0, "exp_142_col5_qwen25_vllm": 165.0,
    "exp_157_paddleocr_titles_squeeze": 5.1, "exp_160_ppdoclayout_headblocks": 6.2,
    "exp_167_grouped": 6.2, "exp_168_grouped_qwen": 150.0,
    "exp_169_merge_045_168": 286.0, "exp_170_asym_merge": 286.0,
    "ensemble_30min": 600.0, "ensemble_prune5": 400.0,
}

DESCRIPTIONS: dict[str, str] = {
    "exp_045_qwen3vl_vllm": "Qwen3-VL structured-JSON OCR over column crops (vllm)",
    "exp_055_col6_ads_prompt": "Qwen3-VL, 6-column split + ads-aware prompt",
    "exp_097_col4_qwen3vl_vllm": "Qwen3-VL, 4-column split",
    "exp_102_fullpage_vllm": "Qwen full-page, no column split — collapses on dense pages",
    "exp_107_fullpage_qwen25vl": "Qwen2.5-VL full-page",
    "exp_138_col4_qwen25_vllm": "Qwen2.5-VL, 4-column split",
    "exp_140_yolo_smallregion_vllm": "DocLayout-YOLO small regions + vllm OCR",
    "exp_142_col5_qwen25_vllm": "Qwen2.5-VL, 5-column split",
    "exp_157_paddleocr_titles_squeeze": "PaddleOCR-VL, YOLO-title headlines + char-run squeeze",
    "exp_160_ppdoclayout_headblocks": "PP-DocLayoutV3 regions + consecutive-title head-block grouping",
}

REFERENCES: dict[str, str] = {
    "ensemble_30min": "8-source recall oracle ensemble (GT-building tool, not production)",
    "ensemble_prune5": "pruned 5-source oracle ensemble",
}


def _exp_num(config: str) -> int:
    m = re.match(r"exp_(\d+)", config)
    return int(m.group(1)) if m else 10_000


def main() -> None:
    rows: list[dict[str, object]] = []
    for config in DESCRIPTIONS:
        results = evaluate_config(config, EVAL_DATES)
        if len(results) != len(EVAL_DATES):
            continue
        score = sum(r.mausoleobench_score for r in results.values()) / len(results)
        cost = GPU_COST.get(config)
        rows.append({"config": config, "exp": _exp_num(config), "score": round(score, 4),
                     "description": DESCRIPTIONS[config], "reference": False,
                     "gpu_s_per_page": cost, "budget_ok": cost is not None and cost <= BUDGET_CAP})
    rows.sort(key=lambda r: r["exp"])
    for i, r in enumerate(rows, 1):
        r["n"] = i

    ref_rows: list[dict[str, object]] = []
    for config, desc in REFERENCES.items():
        results = evaluate_config(config, EVAL_DATES)
        if len(results) != len(EVAL_DATES):
            continue
        score = sum(r.mausoleobench_score for r in results.values()) / len(results)
        ref_rows.append({"config": config, "n": 0, "score": round(score, 4),
                         "description": desc, "name": config.replace("ensemble_", ""), "reference": True,
                         "gpu_s_per_page": GPU_COST.get(config), "budget_ok": False})

    LOG.write_text("\n".join(json.dumps(r) for r in rows + ref_rows) + "\n")
    print(f"seeded {len(rows)} attempts + {len(ref_rows)} references -> {LOG}")


if __name__ == "__main__":
    main()
