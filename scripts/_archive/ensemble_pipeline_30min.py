"""
Production ensemble pipeline (2× RTX 3090, parallel).

Design — 13 sub-pipelines across 2 GPU chains:
  GPU 0: exp_045 → col3_trans → exp_108 → exp_099 → exp_111 → exp_052 → exp_102_fullpage
  GPU 1: exp_010_yolo → exp_055_col6_ads → col4_trans → exp_098 → yolo_qwen25 → exp_107_fullpage_qwen25

Wall-clock: ~50-60 min per issue fresh on 1910 (6 pages). Original 30-min budget
is exceeded — fullpage on 1910 alone takes ~22min, but adding both fullpage variants
brings score from 0.8924 to 0.9088 (~+0.016 — the same as the 17-source unconstrained).

Measured score (cached): 0.9088 avg composite (1885=0.8850, 1910=0.9327).
vs 0.9089 unconstrained 17-config. Effectively matches unconstrained at lower cost.

Usage:
    uv run --no-project python scripts/ensemble_pipeline_30min.py <date> [<date> ...]

Outputs eval/predictions/ensemble_30min_<date>.json.
Sub-pipeline predictions cached via standard run_real_ocr.py paths.
"""
from __future__ import annotations

import json
import os
import pathlib as pl
import subprocess
import sys
import time
import typing as tp

sys.path.insert(0, str(pl.Path(__file__).parent))

from blob_replace_v2 import COL1_SOURCES as CROSSPAGE_COL1_SOURCES, replace_with_pairs
from ensemble_text_replacement import merge_with_replacement
from quality_text_select import select_best_text
from trim_repetitive import trim_predictions


ROOT = pl.Path(__file__).parent.parent
PREDICTIONS_DIR = ROOT / "eval" / "predictions"


GPU0_CHAIN: list[str] = [
    "exp_045_qwen3vl_vllm",
    "col3_qwen3_8b_v2_structured",
    "exp_108_col3_qwen25vl",
    "exp_099_col2_qwen3vl_vllm",
    "exp_111_col2_qwen25vl",
    "exp_052_col6_vllm",
    "exp_102_fullpage_vllm",
]

GPU1_CHAIN: list[str] = [
    "exp_010_yolo_qwen3_8b",
    "exp_055_col6_ads_prompt",
    "col4_qwen3_8b_v2_structured",
    "exp_098_col5_qwen3vl_vllm",
    "yolo_qwen25_7b_v2_structured",
    "exp_107_fullpage_qwen25vl",
    "exp_028_yolo_smallregion",
]


PRIMARY = "exp_045_qwen3vl_vllm"

REPLACEMENT_CHAIN: list[tuple[str, float, float]] = [
    ("col3_qwen3_8b_v2_structured",   0.75, 1.02),
    ("exp_055_col6_ads_prompt",       0.75, 1.05),
    ("exp_010_yolo_qwen3_8b",         0.50, 1.08),
    ("exp_107_fullpage_qwen25vl",     0.50, 1.02),
    ("exp_099_col2_qwen3vl_vllm",     0.75, 1.05),
    ("yolo_qwen25_7b_v2_structured",  0.12, 1.20),
    ("exp_108_col3_qwen25vl",         0.75, 1.05),
    ("exp_028_yolo_smallregion",      0.75, 1.05),
    ("col4_qwen3_8b_v2_structured",   0.85, 1.05),
]

ADDITIVE_SOURCES: list[tuple[str, float, float]] = [
    ("exp_098_col5_qwen3vl_vllm",     0.60, 100.0),
    ("exp_052_col6_vllm",             0.50, 100.0),
    ("exp_102_fullpage_vllm",         0.75, 100.0),
    ("exp_028_yolo_smallregion",      0.85, 100.0),
    ("qwen3b_structured",             0.75, 100.0),
    ("qwen_vl_3b_structured",         0.75, 100.0),
    ("col5_qwen3_8b_v2_structured",   0.50, 100.0),
    ("col3_qwen25_3b_v2_structured",  0.50, 100.0),
    ("exp_111_col2_qwen25vl",         0.85, 100.0),
]

QUALITY_SELECT_SOURCES: list[str] = [
    PRIMARY,
    "exp_055_col6_ads_prompt",
    "exp_010_yolo_qwen3_8b",
]


def _prediction_path(name: str, date: str) -> pl.Path:
    return PREDICTIONS_DIR / f"{name}_{date}.json"


def _load_clean(path: pl.Path) -> dict[str, tp.Any]:
    return trim_predictions(json.load(open(path)))


def _run_chain_on_gpu(chain: list[str], date: str, gpu_index: int, log_path: pl.Path) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["RAY_ADDRESS"] = ""
    cmd = ["uv", "run", "--no-project", "python", str(ROOT / "scripts" / "run_real_ocr.py")]
    for config_name in chain:
        cmd.append(config_name)
    cmd.append(date)
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
        return proc.wait()


def _run_configs_parallel(date: str) -> None:
    missing_gpu0 = [c for c in GPU0_CHAIN if not _prediction_path(c, date).exists()]
    missing_gpu1 = [c for c in GPU1_CHAIN if not _prediction_path(c, date).exists()]
    if not missing_gpu0 and not missing_gpu1:
        print(f"  all sub-pipelines cached for {date}")
        return

    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    gpu0_log = log_dir / f"ensemble_30min_{date}_gpu0.log"
    gpu1_log = log_dir / f"ensemble_30min_{date}_gpu1.log"

    print(f"  GPU0 chain: {missing_gpu0 or 'cached'}")
    print(f"  GPU1 chain: {missing_gpu1 or 'cached'}")

    p0 = None
    p1 = None
    t0 = time.time()
    if missing_gpu0:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["RAY_ADDRESS"] = ""
        cmd = ["uv", "run", "--no-project", "python", str(ROOT / "scripts" / "run_real_ocr.py")]
        for c in missing_gpu0:
            cmd.append(c)
        cmd.append(date)
        p0 = subprocess.Popen(cmd, env=env, stdout=open(gpu0_log, "w"), stderr=subprocess.STDOUT, cwd=str(ROOT))
    if missing_gpu1:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "1"
        env["RAY_ADDRESS"] = ""
        cmd = ["uv", "run", "--no-project", "python", str(ROOT / "scripts" / "run_real_ocr.py")]
        for c in missing_gpu1:
            cmd.append(c)
        cmd.append(date)
        p1 = subprocess.Popen(cmd, env=env, stdout=open(gpu1_log, "w"), stderr=subprocess.STDOUT, cwd=str(ROOT))

    rc0 = p0.wait() if p0 else 0
    rc1 = p1.wait() if p1 else 0
    dt = time.time() - t0
    print(f"  GPU chains finished in {dt:.1f}s (GPU0 rc={rc0}, GPU1 rc={rc1})")
    if rc0 or rc1:
        print(f"  check logs: {gpu0_log} / {gpu1_log}")


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
    col1_predictions = []
    for src in CROSSPAGE_COL1_SOURCES:
        path = _prediction_path(src, date)
        if path.exists():
            col1_predictions.append(json.load(open(path)))
    if col1_predictions:
        current, _, replaced = replace_with_pairs(current, col1_predictions)
        if replaced:
            print(f"  cross-page: replaced {replaced} truncated articles with col1 head+continuation pairs")
    return current


def run(date: str) -> pl.Path:
    wall_start = time.time()
    _run_configs_parallel(date)

    current = _ensemble(date)
    output_path = PREDICTIONS_DIR / f"ensemble_research_{date}.json"
    with open(output_path, "w") as f:
        json.dump(current, f, indent=2, ensure_ascii=False)
    dt = time.time() - wall_start
    print(f"{date}: {len(current['articles'])} articles -> {output_path} ({dt:.1f}s wall)")
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
