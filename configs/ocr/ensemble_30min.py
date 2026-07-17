import pathlib as pl

from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergePages, ParallelEnsembleOcr, ParseIssue, VlmOcr, YoloCrop

CACHE_DIR = str(pl.Path(__file__).resolve().parent.parent.parent / "eval" / "predictions")

QWEN3 = "Qwen/Qwen3-VL-8B-Instruct"
QWEN25 = "Qwen/Qwen2.5-VL-7B-Instruct"


def _vlm_sub(
    name: str,
    model: str,
    *,
    columns: int | None,
    max_model_len: int,
    prompt: str = prompts.VLM_OCR_STRUCTURED_V2,
    strict: bool = False,
    gpu_memory_utilization: float = 0.92,
) -> OcrPipelineConfig:
    split = [ColumnSplit(num_columns=columns, overlap_pct=0.03)] if columns else []
    vlm = VlmOcr(
        model=model,
        prompt=prompt,
        backend="vllm",
        max_tokens=8192,
        max_model_len=max_model_len,
        gpu_fraction=1.0,
        gpu_memory_utilization=gpu_memory_utilization,
        vllm_strict=strict,
    )
    return OcrPipelineConfig(name=name, operators=[*split, vlm, MergePages(), ParseIssue()])


def _yolo_sub(name: str) -> OcrPipelineConfig:
    return OcrPipelineConfig(
        name=name,
        operators=[
            YoloCrop(
                conf_threshold=0.15, gpu_fraction=0.3, min_region_area=1500, merge_vertical_gap=50, merge_horizontal_overlap=0.5, padding=15
            ),
            VlmOcr(
                model=QWEN3,
                prompt=prompts.VLM_OCR_STRUCTURED_V2,
                backend="vllm",
                max_tokens=8192,
                max_model_len=16384,
                max_pixels=24_000_000,
                gpu_fraction=1.0,
                vllm_strict=True,
            ),
            MergePages(),
            ParseIssue(),
        ],
    )


SUB_CONFIGS = (
    _vlm_sub("exp_107_fullpage_qwen25vl", QWEN25, columns=None, max_model_len=20480),
    _vlm_sub("exp_102_fullpage_vllm", QWEN3, columns=None, max_model_len=20480, gpu_memory_utilization=0.94),
    _vlm_sub("exp_055_col6_ads_prompt", QWEN3, columns=6, max_model_len=12288, prompt=prompts.VLM_OCR_ADS_FOCUSED),
    _yolo_sub("exp_140_yolo_smallregion_vllm"),
    _vlm_sub("exp_138_col4_qwen25_vllm", QWEN25, columns=4, max_model_len=16384),
    _vlm_sub("exp_045_qwen3vl_vllm", QWEN3, columns=3, max_model_len=12288),
    _vlm_sub("exp_097_col4_qwen3vl_vllm", QWEN3, columns=4, max_model_len=12288),
    _vlm_sub("exp_142_col5_qwen25_vllm", QWEN25, columns=5, max_model_len=16384),
)


config = OcrPipelineConfig(
    name="ensemble_30min",
    operators=[
        ParallelEnsembleOcr(
            sub_configs=SUB_CONFIGS,
            primary_name="exp_107_fullpage_qwen25vl",
            replacement_chain=(
                ("exp_138_col4_qwen25_vllm", 0.85, 1.05),
                ("exp_045_qwen3vl_vllm", 0.50, 1.05),
                ("exp_055_col6_ads_prompt", 0.30, 1.05),
                ("exp_107_fullpage_qwen25vl", 0.50, 1.02),
                ("exp_138_col4_qwen25_vllm", 0.85, 1.05),
                ("exp_140_yolo_smallregion_vllm", 0.85, 1.02),
                ("exp_102_fullpage_vllm", 0.55, 1.05),
                ("exp_097_col4_qwen3vl_vllm", 0.55, 1.05),
                ("exp_142_col5_qwen25_vllm", 0.85, 1.05),
            ),
            additive_sources=(("exp_055_col6_ads_prompt", 0.88, 100.0),),
            quality_select_sources=(
                "exp_045_qwen3vl_vllm",
                "exp_107_fullpage_qwen25vl",
                "exp_138_col4_qwen25_vllm",
                "exp_055_col6_ads_prompt",
            ),
            crosspage_col1_sources=(),
            min_quality_delta=0.10,
            headline_delta=0.15,
            cache_dir=CACHE_DIR,
            num_gpus=2,
            sub_timeout_s=10800,
        ),
        ParseIssue(),
    ],
)
