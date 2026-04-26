"""Single-config ensemble pipeline reproducing the 0.9087 avg composite baseline.

One OcrPipelineConfig. One run. All 16 sub-pipelines executed internally,
ensembled, trimmed, and emitted as a single Issue.

Usage:
    uv run --no-project python scripts/run_real_ocr.py ensemble_best 1885-06-15

Caches each sub-pipeline prediction to eval/predictions/<name>_<date>.json —
re-runs skip cached work. Fresh runtime ~2-3h per issue on 2x RTX 3090.
Violates the 30-min/issue budget; use configs/ocr/ensemble_30min.py or
scripts/ensemble_pipeline_30min.py for the budget-constrained variant (0.8824).
"""
from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import (
    ColumnSplit,
    EnsembleOcr,
    MergePages,
    ParseIssue,
    VlmOcr,
    YoloCrop,
)


def _col_vlm(name, n, prompt, backend, max_model_len=12288):
    return OcrPipelineConfig(
        name=name,
        operators=[
            ColumnSplit(num_columns=n, overlap_pct=0.03),
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


def _yolo_vlm(name, model):
    return OcrPipelineConfig(
        name=name,
        operators=[
            YoloCrop(conf_threshold=0.2, gpu_fraction=0.3, min_region_area=3000),
            VlmOcr(model=model, prompt=prompts.VLM_OCR_STRUCTURED_V2, backend="transformers", max_tokens=8192, max_model_len=32768),
            MergePages(),
            ParseIssue(),
        ],
    )


def _fullpage(name, model, prompt=prompts.VLM_OCR_STRUCTURED_V2):
    return OcrPipelineConfig(
        name=name,
        operators=[
            VlmOcr(model=model, prompt=prompt, backend="vllm", max_tokens=8192, max_model_len=20480, gpu_fraction=1.0),
            MergePages(),
            ParseIssue(),
        ],
    )


def _col_qwen25(name, n, max_model_len=12288):
    return OcrPipelineConfig(
        name=name,
        operators=[
            ColumnSplit(num_columns=n, overlap_pct=0.03),
            VlmOcr(
                model="Qwen/Qwen2.5-VL-7B-Instruct",
                prompt=prompts.VLM_OCR_STRUCTURED_V2,
                backend="vllm",
                max_tokens=8192,
                max_model_len=max_model_len,
                gpu_fraction=1.0,
            ),
            MergePages(),
            ParseIssue(),
        ],
    )


SUB_PIPELINES = (
    _col_vlm("exp_045_qwen3vl_vllm",           3, prompts.VLM_OCR_STRUCTURED_V2, "vllm"),
    _col_vlm("col3_qwen3_8b_v2_structured",    3, prompts.VLM_OCR_STRUCTURED_V2, "transformers"),
    _col_vlm("exp_055_col6_ads_prompt",        6, prompts.VLM_OCR_ADS_FOCUSED,   "vllm"),
    _yolo_vlm("exp_010_yolo_qwen3_8b",         "Qwen/Qwen3-VL-8B-Instruct"),
    _col_vlm("col4_qwen3_8b_v2_structured",    4, prompts.VLM_OCR_STRUCTURED_V2, "transformers"),
    _col_vlm("exp_097_col4_qwen3vl_vllm",      4, prompts.VLM_OCR_STRUCTURED_V2, "vllm"),
    _col_vlm("col5_qwen3_8b_v2_structured",    5, prompts.VLM_OCR_STRUCTURED_V2, "transformers"),
    _col_vlm("exp_052_col6_vllm",              6, prompts.VLM_OCR_STRUCTURED_V2, "vllm"),
    _yolo_vlm("yolo_qwen25_7b_v2_structured",  "Qwen/Qwen2.5-VL-7B-Instruct"),
    _col_vlm("exp_098_col5_qwen3vl_vllm",      5, prompts.VLM_OCR_STRUCTURED_V2, "vllm"),
    _col_vlm("exp_099_col2_qwen3vl_vllm",      2, prompts.VLM_OCR_STRUCTURED_V2, "vllm", max_model_len=20480),
    _fullpage("exp_102_fullpage_vllm",         "Qwen/Qwen3-VL-8B-Instruct"),
    _col_vlm("exp_105_col1_qwen3vl_vllm",      1, prompts.VLM_OCR_STRUCTURED_V2, "vllm", max_model_len=20480),
    _fullpage("exp_107_fullpage_qwen25vl",     "Qwen/Qwen2.5-VL-7B-Instruct"),
    _col_qwen25("exp_108_col3_qwen25vl",       3),
    _col_qwen25("exp_109_col4_qwen25vl",       4),
    _col_qwen25("exp_111_col2_qwen25vl",       2, max_model_len=20480),
    OcrPipelineConfig(
        name="exp_125_yolo_smallconf_vllm",
        operators=[
            YoloCrop(conf_threshold=0.15, gpu_fraction=0.3, min_region_area=1500),
            VlmOcr(
                model="Qwen/Qwen3-VL-8B-Instruct",
                prompt=prompts.VLM_OCR_STRUCTURED_V2,
                backend="vllm",
                max_tokens=2048,
                max_model_len=13312,
                gpu_fraction=1.0,
            ),
            MergePages(),
            ParseIssue(),
        ],
    ),
)


config = OcrPipelineConfig(
    name="ensemble_best",
    operators=[
        EnsembleOcr(
            sub_configs=SUB_PIPELINES,
            primary_name="exp_045_qwen3vl_vllm",
            replacement_chain=(
                ("col3_qwen3_8b_v2_structured",   0.75, 1.15),
                ("exp_055_col6_ads_prompt",       0.75, 1.08),
                ("exp_010_yolo_qwen3_8b",         0.50, 1.08),
                ("col4_qwen3_8b_v2_structured",   0.75, 1.02),
                ("exp_097_col4_qwen3vl_vllm",     0.75, 1.02),
            ),
            additive_sources=(
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
                ("exp_125_yolo_smallconf_vllm",   0.75, 100.0),
            ),
            quality_select_sources=(
                "exp_045_qwen3vl_vllm",
                "col3_qwen3_8b_v2_structured",
                "exp_055_col6_ads_prompt",
                "exp_010_yolo_qwen3_8b",
                "col4_qwen3_8b_v2_structured",
                "col5_qwen3_8b_v2_structured",
                "exp_052_col6_vllm",
            ),
            min_quality_delta=0.10,
            headline_delta=0.15,
        ),
        ParseIssue(),
    ],
)
