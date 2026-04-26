"""
Single-file reproducible pipeline for the **0.9087 avg composite** baseline.

Defines 16 OcrPipelineConfig sub-pipelines inline, runs each sequentially
(caching to disk), then applies the ensemble merge + JSON-blob filter +
quality_text_select post-processing.

Usage:
    uv run --no-project python scripts/ensemble_pipeline.py <issue_date> [<issue_date> ...]

    issue_date must match a directory under eval/ground_truth/<date>/
    containing numbered .jpeg page images.

Output: eval/predictions/ensemble_best_<date>.json

Runtime: ~2-3h per issue on 2x RTX 3090 (fresh, no cache). Sub-pipeline
results are cached to eval/predictions/<config>_<date>.json; re-runs are
seconds when cached. Violates the 30-min/issue production budget —
kept as the research upper bound.

For a budget-constrained pipeline (0.8824 in ≤30 min), use
scripts/ensemble_pipeline_30min.py.
"""
from __future__ import annotations

import json
import pathlib as pl
import sys
import typing as tp

sys.path.insert(0, str(pl.Path(__file__).parent))

from ensemble_text_replacement import merge_with_replacement
from quality_text_select import select_best_text
from trim_repetitive import trim_predictions

from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergePages, ParseIssue, VlmOcr, YoloCrop


ROOT = pl.Path(__file__).parent.parent
PREDICTIONS_DIR = ROOT / "eval" / "predictions"
GROUND_TRUTH_DIR = ROOT / "eval" / "ground_truth"


def _col_vlm(name: str, num_columns: int, prompt: str, backend: str, max_model_len: int = 12288) -> OcrPipelineConfig:
    return OcrPipelineConfig(
        name=name,
        operators=[
            ColumnSplit(num_columns=num_columns, overlap_pct=0.03),
            VlmOcr(
                model="Qwen/Qwen3-VL-8B-Instruct",
                prompt=prompt,
                backend=backend,
                max_tokens=8192,
                max_model_len=max_model_len,
                gpu_fraction=1.0,
            ),
            MergePages(),
            ParseIssue(),
        ],
    )


def _yolo_vlm(name: str, model: str) -> OcrPipelineConfig:
    return OcrPipelineConfig(
        name=name,
        operators=[
            YoloCrop(conf_threshold=0.2, gpu_fraction=0.3, min_region_area=3000),
            VlmOcr(
                model=model,
                prompt=prompts.VLM_OCR_STRUCTURED_V2,
                backend="transformers",
                max_tokens=8192,
                max_model_len=32768,
            ),
            MergePages(),
            ParseIssue(),
        ],
    )


SUB_PIPELINES: dict[str, OcrPipelineConfig] = {
    "exp_045_qwen3vl_vllm":           _col_vlm("exp_045_qwen3vl_vllm",           3, prompts.VLM_OCR_STRUCTURED_V2, "vllm"),
    "col3_qwen3_8b_v2_structured":    _col_vlm("col3_qwen3_8b_v2_structured",    3, prompts.VLM_OCR_STRUCTURED_V2, "transformers"),
    "exp_055_col6_ads_prompt":        _col_vlm("exp_055_col6_ads_prompt",        6, prompts.VLM_OCR_ADS_FOCUSED,   "vllm"),
    "exp_010_yolo_qwen3_8b":          _yolo_vlm("exp_010_yolo_qwen3_8b",         "Qwen/Qwen3-VL-8B-Instruct"),
    "col4_qwen3_8b_v2_structured":    _col_vlm("col4_qwen3_8b_v2_structured",    4, prompts.VLM_OCR_STRUCTURED_V2, "transformers"),
    "exp_097_col4_qwen3vl_vllm":      _col_vlm("exp_097_col4_qwen3vl_vllm",      4, prompts.VLM_OCR_STRUCTURED_V2, "vllm"),
    "col5_qwen3_8b_v2_structured":    _col_vlm("col5_qwen3_8b_v2_structured",    5, prompts.VLM_OCR_STRUCTURED_V2, "transformers"),
    "exp_052_col6_vllm":              _col_vlm("exp_052_col6_vllm",              6, prompts.VLM_OCR_STRUCTURED_V2, "vllm"),
    "yolo_qwen25_7b_v2_structured":   _yolo_vlm("yolo_qwen25_7b_v2_structured",  "Qwen/Qwen2.5-VL-7B-Instruct"),
    "exp_098_col5_qwen3vl_vllm":      _col_vlm("exp_098_col5_qwen3vl_vllm",      5, prompts.VLM_OCR_STRUCTURED_V2, "vllm"),
    "exp_099_col2_qwen3vl_vllm":      _col_vlm("exp_099_col2_qwen3vl_vllm",      2, prompts.VLM_OCR_STRUCTURED_V2, "vllm", max_model_len=20480),
    "exp_102_fullpage_vllm":          OcrPipelineConfig(
        name="exp_102_fullpage_vllm",
        operators=[
            VlmOcr(
                model="Qwen/Qwen3-VL-8B-Instruct",
                prompt=prompts.VLM_OCR_STRUCTURED_V2,
                backend="vllm",
                max_tokens=8192,
                max_model_len=20480,
                gpu_fraction=1.0,
            ),
            MergePages(),
            ParseIssue(),
        ],
    ),
    "exp_105_col1_qwen3vl_vllm":      _col_vlm("exp_105_col1_qwen3vl_vllm",      1, prompts.VLM_OCR_STRUCTURED_V2, "vllm", max_model_len=20480),
    "exp_107_fullpage_qwen25vl":      OcrPipelineConfig(
        name="exp_107_fullpage_qwen25vl",
        operators=[
            VlmOcr(
                model="Qwen/Qwen2.5-VL-7B-Instruct",
                prompt=prompts.VLM_OCR_STRUCTURED_V2,
                backend="vllm",
                max_tokens=8192,
                max_model_len=20480,
                gpu_fraction=1.0,
            ),
            MergePages(),
            ParseIssue(),
        ],
    ),
    "exp_108_col3_qwen25vl":          OcrPipelineConfig(
        name="exp_108_col3_qwen25vl",
        operators=[
            ColumnSplit(num_columns=3, overlap_pct=0.03),
            VlmOcr(model="Qwen/Qwen2.5-VL-7B-Instruct", prompt=prompts.VLM_OCR_STRUCTURED_V2, backend="vllm", max_tokens=8192, max_model_len=12288, gpu_fraction=1.0),
            MergePages(),
            ParseIssue(),
        ],
    ),
    "exp_109_col4_qwen25vl":          OcrPipelineConfig(
        name="exp_109_col4_qwen25vl",
        operators=[
            ColumnSplit(num_columns=4, overlap_pct=0.03),
            VlmOcr(model="Qwen/Qwen2.5-VL-7B-Instruct", prompt=prompts.VLM_OCR_STRUCTURED_V2, backend="vllm", max_tokens=8192, max_model_len=12288, gpu_fraction=1.0),
            MergePages(),
            ParseIssue(),
        ],
    ),
    "exp_111_col2_qwen25vl":          OcrPipelineConfig(
        name="exp_111_col2_qwen25vl",
        operators=[
            ColumnSplit(num_columns=2, overlap_pct=0.03),
            VlmOcr(model="Qwen/Qwen2.5-VL-7B-Instruct", prompt=prompts.VLM_OCR_STRUCTURED_V2, backend="vllm", max_tokens=8192, max_model_len=20480, gpu_fraction=1.0),
            MergePages(),
            ParseIssue(),
        ],
    ),
}


PRIMARY = "exp_045_qwen3vl_vllm"

REPLACEMENT_CHAIN: list[tuple[str, float, float]] = [
    ("col3_qwen3_8b_v2_structured",   0.75, 1.15),
    ("exp_055_col6_ads_prompt",       0.75, 1.08),
    ("exp_010_yolo_qwen3_8b",         0.50, 1.08),
    ("col4_qwen3_8b_v2_structured",   0.75, 1.02),
    ("exp_097_col4_qwen3vl_vllm",     0.75, 1.02),
]

ADDITIVE_SOURCES: list[tuple[str, float, float]] = [
    ("col5_qwen3_8b_v2_structured",   0.50, 100.0),
    ("exp_052_col6_vllm",             0.30, 100.0),
    ("yolo_qwen25_7b_v2_structured",  0.50, 100.0),
    ("exp_098_col5_qwen3vl_vllm",     0.50, 100.0),
    ("exp_099_col2_qwen3vl_vllm",     0.75, 100.0),
    ("exp_102_fullpage_vllm",         0.75, 100.0),
    ("exp_105_col1_qwen3vl_vllm",     0.75, 100.0),
    ("exp_107_fullpage_qwen25vl",     0.75, 100.0),
    ("exp_108_col3_qwen25vl",         0.75, 100.0),
    ("exp_109_col4_qwen25vl",         0.75, 100.0),
    ("exp_111_col2_qwen25vl",         0.75, 100.0),
]

QUALITY_SELECT_SOURCES: list[str] = [
    PRIMARY,
    "col3_qwen3_8b_v2_structured",
    "exp_055_col6_ads_prompt",
    "exp_010_yolo_qwen3_8b",
    "col4_qwen3_8b_v2_structured",
    "col5_qwen3_8b_v2_structured",
    "exp_052_col6_vllm",
]


def _prediction_path(name: str, date: str) -> pl.Path:
    return PREDICTIONS_DIR / f"{name}_{date}.json"


def _load_clean(path: pl.Path) -> dict[str, tp.Any]:
    return trim_predictions(json.load(open(path)))


def _run_sub_pipeline(name: str, config: OcrPipelineConfig, date: str) -> None:
    from run_real_ocr import load_issue_images
    from mausoleo.ocr.pipeline import run_pipeline

    output_path = _prediction_path(name, date)
    if output_path.exists():
        print(f"  [cached] {name} {date}")
        return

    issue_dir = GROUND_TRUTH_DIR / date
    images = load_issue_images(issue_dir)
    print(f"  [run]    {name} {date} ({len(images)} pages)")
    result = run_pipeline(config, images)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def _ensemble(date: str) -> dict[str, tp.Any]:
    current = _load_clean(_prediction_path(PRIMARY, date))

    for source, overlap, ratio in REPLACEMENT_CHAIN:
        extra = _load_clean(_prediction_path(source, date))
        current = merge_with_replacement(current, extra, overlap_threshold=overlap, replace_ratio=ratio)

    for source, overlap, ratio in ADDITIVE_SOURCES:
        extra = _load_clean(_prediction_path(source, date))
        current = merge_with_replacement(current, extra, overlap_threshold=overlap, replace_ratio=ratio)

    quality_sources = [json.load(open(_prediction_path(s, date))) for s in QUALITY_SELECT_SOURCES]
    current = select_best_text(current, quality_sources, min_quality_delta=0.10, headline_delta=0.15)
    current = trim_predictions(current)
    return current


def run(date: str) -> pl.Path:
    print(f"Running 11 sub-pipelines for {date}...")
    for name, config in SUB_PIPELINES.items():
        _run_sub_pipeline(name, config, date)

    print(f"Building ensemble...")
    result = _ensemble(date)

    output_path = PREDICTIONS_DIR / f"ensemble_best_{date}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"{date}: {len(result['articles'])} articles -> {output_path}")
    return output_path


def main() -> None:
    if len(sys.argv) < 2:
        dates = ["1885-06-15", "1910-06-15"]
    else:
        dates = sys.argv[1:]
    for date in dates:
        run(date)


if __name__ == "__main__":
    main()
