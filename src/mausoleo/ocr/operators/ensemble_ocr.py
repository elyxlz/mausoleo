"""Ensemble operator: runs multiple sub-pipelines serially and merges their outputs.

One OcrPipelineConfig can contain an EnsembleOcr op that internally orchestrates
all column-split/yolo/VLM variants, then applies trim + merge + quality_text_select.
"""
from __future__ import annotations

import base64
import dataclasses as dc
import json
import sys
import typing as tp

from mausoleo.ocr.operators.base import BaseOperatorConfig, OperatorType, register_operator


@dc.dataclass(frozen=True, kw_only=True)
class EnsembleOcr(BaseOperatorConfig):
    sub_configs: tuple[tp.Any, ...] = ()
    primary_name: str = ""
    replacement_chain: tuple[tuple[str, float, float], ...] = ()
    additive_sources: tuple[tuple[str, float, float], ...] = ()
    quality_select_sources: tuple[str, ...] = ()
    min_quality_delta: float = 0.10
    headline_delta: float = 0.15
    cache_dir: str = "eval/predictions"
    cache_date: str = ""


@register_operator(EnsembleOcr, operation=OperatorType.MAP)
def _ensemble_ocr(row: dict[str, tp.Any], *, config: EnsembleOcr) -> dict[str, tp.Any]:
    import pathlib as pl

    from mausoleo.ocr.merge import (
        merge_with_replacement,
        select_best_text,
        trim_predictions,
    )

    from mausoleo.ocr.pipeline import run_pipeline

    date = config.cache_date or str(row.get("date", ""))
    cache = pl.Path(config.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    images = [base64.b64decode(b64) for b64 in str(row["images_b64"]).split("|")]
    predictions: dict[str, dict[str, tp.Any]] = {}
    for sub in config.sub_configs:
        name = sub.name
        out_path = cache / f"{name}_{date}.json"
        if out_path.exists():
            predictions[name] = json.loads(out_path.read_text())
            continue
        issue = run_pipeline(sub, images, date=date)
        pred = dc.asdict(issue)
        out_path.write_text(json.dumps(pred, indent=2, ensure_ascii=False))
        predictions[name] = pred

    def load(name: str) -> dict[str, tp.Any]:
        return trim_predictions(predictions[name])

    current = load(config.primary_name)
    for src, ov, rt in config.replacement_chain:
        if src not in predictions:
            continue
        current = merge_with_replacement(current, load(src), overlap_threshold=ov, replace_ratio=rt)
    for src, ov, rt in config.additive_sources:
        if src not in predictions:
            continue
        current = merge_with_replacement(current, load(src), overlap_threshold=ov, replace_ratio=rt)

    qs_list: list[dict[str, tp.Any]] = []
    for name in config.quality_select_sources:
        if name in predictions:
            qs_list.append(predictions[name])
    current = select_best_text(current, qs_list, min_quality_delta=config.min_quality_delta, headline_delta=config.headline_delta)
    current = trim_predictions(current)

    return {**row, "result_json": json.dumps(current, ensure_ascii=False)}
