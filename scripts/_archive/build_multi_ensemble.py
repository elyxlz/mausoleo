from __future__ import annotations

import json
import sys

from ensemble_text_replacement import merge_with_replacement
from quality_text_select import select_best_text
from trim_repetitive import trim_predictions


PRIMARY = "exp_045_qwen3vl_vllm"

REPLACEMENT_CHAIN = [
    ("col3_qwen3_8b_v2_structured", 0.75, 1.15),
    ("exp_055_col6_ads_prompt", 0.75, 1.08),
    ("exp_010_yolo_qwen3_8b", 0.50, 1.08),
    ("col4_qwen3_8b_v2_structured", 0.75, 1.02),
    ("exp_097_col4_qwen3vl_vllm", 0.75, 1.02),
]

ADDITIVE_SOURCES = [
    ("col5_qwen3_8b_v2_structured", 0.50, 100.0),
    ("exp_052_col6_vllm", 0.30, 100.0),
    ("yolo_qwen25_7b_v2_structured", 0.50, 100.0),
    ("exp_098_col5_qwen3vl_vllm", 0.50, 100.0),
    ("exp_099_col2_qwen3vl_vllm", 0.75, 100.0),
    ("exp_102_fullpage_vllm", 0.75, 100.0),
    ("exp_105_col1_qwen3vl_vllm", 0.75, 100.0),
    ("exp_107_fullpage_qwen25vl", 0.75, 100.0),
    ("exp_108_col3_qwen25vl", 0.75, 100.0),
    ("exp_109_col4_qwen25vl", 0.75, 100.0),
    ("exp_111_col2_qwen25vl", 0.75, 100.0),
]

QUALITY_SELECT_SOURCES = [
    PRIMARY,
    "exp_055_col6_ads_prompt",
    "exp_010_yolo_qwen3_8b",
    "col4_qwen3_8b_v2_structured",
    "col5_qwen3_8b_v2_structured",
    "exp_052_col6_vllm",
    "exp_045_qwen3vl_vllm",
    "exp_056_col4_ads_prompt",
]


def _load_cleaned(path: str) -> dict:
    return trim_predictions(json.load(open(path)))


def build_ensemble(date: str, output_path: str) -> None:
    primary_path = f"eval/predictions/{PRIMARY}_{date}.json"
    current = _load_cleaned(primary_path)

    for source, overlap, ratio in REPLACEMENT_CHAIN:
        path = f"eval/predictions/{source}_{date}.json"
        try:
            extra = _load_cleaned(path)
        except FileNotFoundError:
            print(f"  skip missing (replace): {path}")
            continue
        current = merge_with_replacement(current, extra, overlap_threshold=overlap, replace_ratio=ratio)

    for source, overlap, ratio in ADDITIVE_SOURCES:
        path = f"eval/predictions/{source}_{date}.json"
        try:
            extra = _load_cleaned(path)
        except FileNotFoundError:
            print(f"  skip missing (additive): {path}")
            continue
        current = merge_with_replacement(current, extra, overlap_threshold=overlap, replace_ratio=ratio)

    quality_sources: list[dict] = []
    for source in QUALITY_SELECT_SOURCES:
        path = f"eval/predictions/{source}_{date}.json"
        try:
            quality_sources.append(json.load(open(path)))
        except FileNotFoundError:
            continue
    current = select_best_text(current, quality_sources, min_quality_delta=0.10, headline_delta=0.15)
    current = trim_predictions(current)

    with open(output_path, "w") as f:
        json.dump(current, f, indent=2, ensure_ascii=False)
    print(f"{date}: {len(current['articles'])} articles saved to {output_path}")


def main() -> None:
    if len(sys.argv) < 2:
        for date in ["1885-06-15", "1910-06-15"]:
            build_ensemble(date, f"eval/predictions/ensemble_best_{date}.json")
    else:
        date = sys.argv[1]
        build_ensemble(date, f"eval/predictions/ensemble_best_{date}.json")


if __name__ == "__main__":
    main()
