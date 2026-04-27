from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ParallelEnsembleOcr, ParseIssue


GPU0_CHAIN = (
    "exp_107_fullpage_qwen25vl",
    "exp_045_qwen3vl_vllm",
    "exp_055_col6_ads_prompt",
)

GPU1_CHAIN = (
    "exp_134_yolo_vllm_strict",
    "exp_138_col4_qwen25_vllm",
    "exp_140_yolo_smallregion_vllm",
    "exp_142_col5_qwen25_vllm",
)


config = OcrPipelineConfig(
    name="ensemble_30min",
    operators=[
        ParallelEnsembleOcr(
            gpu0_chain=GPU0_CHAIN,
            gpu1_chain=GPU1_CHAIN,
            primary_name="exp_107_fullpage_qwen25vl",
            replacement_chain=(
                ("exp_045_qwen3vl_vllm",            0.50, 1.02),
                ("exp_138_col4_qwen25_vllm",        0.85, 1.05),
                ("exp_055_col6_ads_prompt",         0.30, 1.02),
                ("exp_134_yolo_vllm_strict",        0.25, 1.10),
                ("exp_107_fullpage_qwen25vl",       0.50, 1.02),
                ("exp_142_col5_qwen25_vllm",        0.85, 1.05),
                ("exp_140_yolo_smallregion_vllm",   0.85, 1.05),
            ),
            additive_sources=(
                ("exp_055_col6_ads_prompt",        0.88, 100.0),
                ("exp_140_yolo_smallregion_vllm",  0.85, 100.0),
                ("exp_142_col5_qwen25_vllm",       0.85, 100.0),
            ),
            quality_select_sources=(
                "exp_045_qwen3vl_vllm",
                "exp_107_fullpage_qwen25vl",
                "exp_138_col4_qwen25_vllm",
                "exp_055_col6_ads_prompt",
                "exp_134_yolo_vllm_strict",
            ),
            crosspage_col1_sources=(),
            min_quality_delta=0.10,
            headline_delta=0.15,
        ),
        ParseIssue(),
    ],
)
