from __future__ import annotations

import base64
import json
import typing as tp

import ray
import ray.data

from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.models import Issue, issue_from_dict
from mausoleo.ocr.operators.base import apply_operator


def encode_images(images: list[bytes]) -> str:
    return "|".join(base64.b64encode(img).decode() for img in images)


def setup_ray(n_gpu_operators: int = 1) -> int:
    if not ray.is_initialized():
        import os
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        cudnn_lib = os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "cudnn", "lib")
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        propagated_ld = f"{torch_lib}:{cudnn_lib}:{existing_ld}"
        ray.init(
            ignore_reinit_error=True,
            runtime_env={
                "excludes": ["pyproject.toml", "uv.lock", ".venv", ".git"],
                "env_vars": {"LD_LIBRARY_PATH": propagated_ld},
            },
        )
    n_gpu = int(ray.available_resources().get("GPU", 0))
    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.preserve_order = True
    if n_gpu > 0 and n_gpu_operators > 1:
        ctx.execution_options.resource_limits = ray.data.ExecutionResources(gpu=n_gpu)
    return n_gpu


def run_pipeline(
    config: OcrPipelineConfig,
    images: list[bytes],
    date: str,
    source: str = "il_messaggero",
) -> Issue:
    n_gpu_operators = sum(1 for op in config.operators if op.gpu_fraction > 0)
    n_gpu = setup_ray(n_gpu_operators)

    ds = ray.data.from_items(
        [
            {
                "issue_id": date,
                "date": date,
                "source": source,
                "page_count": len(images),
                "images_b64": encode_images(images),
            }
        ]
    )

    for op_config in config.operators:
        ds = apply_operator(ds, step_config=op_config, n_gpu=n_gpu, n_gpu_operators=n_gpu_operators)

    rows: list[dict[str, tp.Any]] = ds.take(1)
    if not rows:
        raise RuntimeError("pipeline produced no output")

    result = issue_from_dict(json.loads(rows[0]["issue_json"]))
    ray.shutdown()
    return result
