from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ParallelEnsembleOcr, ParseIssue


GPU0_CHAIN = (
    "exp_107_fullpage_qwen25vl",
    "exp_045_qwen3vl_vllm",
    "exp_055_col6_ads_prompt",
)

GPU1_CHAIN = (
    "exp_010_yolo_qwen3_8b",
    "col4_qwen3_8b_v2_structured",
)


config = OcrPipelineConfig(
    name="ensemble_30min",
    operators=[
        ParallelEnsembleOcr(
            gpu0_chain=GPU0_CHAIN,
            gpu1_chain=GPU1_CHAIN,
            primary_name="exp_045_qwen3vl_vllm",
            replacement_chain=(
                ("col3_qwen3_8b_v2_structured", 0.75, 1.02),
                ("exp_052_col6_vllm",           0.75, 1.05),
                ("exp_055_col6_ads_prompt",     0.75, 1.03),
                ("exp_107_fullpage_qwen25vl",   0.50, 1.02),
                ("exp_010_yolo_qwen3_8b",       0.45, 1.10),
                ("exp_099_col2_qwen3vl_vllm",   0.75, 1.05),
                ("exp_125_yolo_smallconf_vllm", 0.50, 1.10),
                ("yolo_qwen25_7b_v2_structured", 0.12, 1.20),
                ("exp_108_col3_qwen25vl",       0.75, 1.05),
                ("exp_014_fullpage",            0.50, 1.05),
                ("exp_028_yolo_smallregion",    0.80, 1.10),
                ("col4_qwen3_8b_v2_structured", 0.85, 1.05),
                ("exp_113_col1_qwen25vl",       0.50, 1.05),
                ("exp_127_yolo_ads_vllm",       0.85, 1.05),
                ("exp_109_col4_qwen25vl",       0.85, 1.05),
                ("exp_028_yolo_smallregion",    0.80, 1.05),
                ("exp_107_fullpage_qwen25vl",   0.50, 1.02),
            ),
            additive_sources=(
                ("exp_111_col2_qwen25vl",       0.85, 100.0),
                ("exp_105_col1_qwen3vl_vllm",   0.50, 100.0),
                ("exp_098_col5_qwen3vl_vllm",   0.60, 100.0),
                ("exp_052_col6_vllm",           0.50, 100.0),
                ("qwen3b_structured",           0.75, 100.0),
                ("exp_028_yolo_smallregion",    0.85, 100.0),
                ("col3_qwen25_3b_v2_structured", 0.50, 100.0),
            ),
            quality_select_sources=(
                "exp_045_qwen3vl_vllm",
                "exp_055_col6_ads_prompt",
                "exp_010_yolo_qwen3_8b",
            ),
            crosspage_col1_sources=(
                "exp_106_col1_ads_vllm",
            ),
            min_quality_delta=0.10,
            headline_delta=0.15,
        ),
        ParseIssue(),
    ],
)
