import pathlib as pl

from mausoleo.ocr import prompts
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.operators import ColumnSplit, MergePages, ParallelEnsembleOcr, ParseIssue, VlmOcr

CACHE_DIR = str(pl.Path(__file__).resolve().parent.parent.parent / "eval" / "predictions")

QWEN3 = "Qwen/Qwen3-VL-8B-Instruct"
QWEN25 = "Qwen/Qwen2.5-VL-7B-Instruct"


def _vlm_sub(
    name: str,
    model: str,
    *,
    columns: int | None,
    max_model_len: int,
    gpu_memory_utilization: float = 0.92,
) -> OcrPipelineConfig:
    split = [ColumnSplit(num_columns=columns, overlap_pct=0.03)] if columns else []
    vlm = VlmOcr(
        model=model,
        prompt=prompts.VLM_OCR_STRUCTURED_V2,
        backend="vllm",
        max_tokens=8192,
        max_model_len=max_model_len,
        gpu_fraction=1.0,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    return OcrPipelineConfig(name=name, operators=[*split, vlm, MergePages(), ParseIssue()])


SUB_CONFIGS = (
    _vlm_sub("exp_107_fullpage_qwen25vl", QWEN25, columns=None, max_model_len=20480),
    _vlm_sub("exp_102_fullpage_vllm", QWEN3, columns=None, max_model_len=20480, gpu_memory_utilization=0.94),
    _vlm_sub("exp_138_col4_qwen25_vllm", QWEN25, columns=4, max_model_len=16384),
    _vlm_sub("exp_045_qwen3vl_vllm", QWEN3, columns=3, max_model_len=12288),
    _vlm_sub("exp_097_col4_qwen3vl_vllm", QWEN3, columns=4, max_model_len=12288),
)


config = OcrPipelineConfig(
    name="ensemble_prune5",
    operators=[
        ParallelEnsembleOcr(
            sub_configs=SUB_CONFIGS,
            primary_name="exp_107_fullpage_qwen25vl",
            replacement_chain=(
                ("exp_138_col4_qwen25_vllm", 0.85, 1.05),
                ("exp_045_qwen3vl_vllm", 0.50, 1.05),
                ("exp_107_fullpage_qwen25vl", 0.50, 1.02),
                ("exp_138_col4_qwen25_vllm", 0.85, 1.05),
                ("exp_102_fullpage_vllm", 0.55, 1.05),
                ("exp_097_col4_qwen3vl_vllm", 0.55, 1.05),
            ),
            additive_sources=(),
            quality_select_sources=(
                "exp_045_qwen3vl_vllm",
                "exp_107_fullpage_qwen25vl",
                "exp_138_col4_qwen25_vllm",
            ),
            crosspage_col1_sources=(),
            min_quality_delta=0.10,
            headline_delta=0.15,
            cache_dir=CACHE_DIR,
            num_gpus=2,
        ),
        ParseIssue(),
    ],
)
