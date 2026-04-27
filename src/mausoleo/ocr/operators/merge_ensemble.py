from __future__ import annotations

import dataclasses as dc
import json
import sys
import typing as tp

from mausoleo.ocr.operators.base import BaseOperatorConfig, OperatorType, register_operator


@dc.dataclass(frozen=True, kw_only=True)
class MergeEnsemble(BaseOperatorConfig):
    primary: str = ""
    replacement_chain: tuple[tuple[str, float, float], ...] = ()
    additive_sources: tuple[tuple[str, float, float], ...] = ()
    quality_select_sources: tuple[str, ...] = ()
    crosspage_col1_sources: tuple[str, ...] = ()
    min_quality_delta: float = 0.10
    headline_delta: float = 0.15


def _import_merge_helpers() -> tuple[tp.Callable[..., tp.Any], ...]:
    from mausoleo.ocr.merge import (
        merge_with_replacement,
        replace_with_pairs,
        select_best_text,
        trim_predictions,
    )
    return trim_predictions, merge_with_replacement, select_best_text, replace_with_pairs


@register_operator(MergeEnsemble, operation=OperatorType.MAP)
def merge_ensemble(row: dict[str, tp.Any], *, config: MergeEnsemble) -> dict[str, tp.Any]:
    trim_predictions, merge_with_replacement, select_best_text, replace_with_pairs = _import_merge_helpers()

    def load_clean(name: str) -> dict[str, tp.Any] | None:
        raw = row.get(name)
        if raw is None:
            return None
        return trim_predictions(json.loads(raw))

    def load_raw(name: str) -> dict[str, tp.Any] | None:
        raw = row.get(name)
        if raw is None:
            return None
        return json.loads(raw)

    primary = load_clean(config.primary)
    if primary is None:
        return {**row, "result_json": json.dumps({"articles": []})}

    current = primary
    for src, ov, rt in config.replacement_chain:
        extra = load_clean(src)
        if extra is None:
            continue
        current = merge_with_replacement(current, extra, overlap_threshold=ov, replace_ratio=rt)
    for src, ov, rt in config.additive_sources:
        extra = load_clean(src)
        if extra is None:
            continue
        current = merge_with_replacement(current, extra, overlap_threshold=ov, replace_ratio=rt)

    qs_list: list[dict[str, tp.Any]] = []
    for name in config.quality_select_sources:
        raw = load_raw(name)
        if raw is not None:
            qs_list.append(raw)
    current = select_best_text(
        current,
        qs_list,
        min_quality_delta=config.min_quality_delta,
        headline_delta=config.headline_delta,
    )
    current = trim_predictions(current)

    col1: list[dict[str, tp.Any]] = []
    for name in config.crosspage_col1_sources:
        raw = load_raw(name)
        if raw is not None:
            col1.append(raw)
    if col1:
        current, _, _ = replace_with_pairs(current, col1)

    return {**row, "result_json": json.dumps(current, ensure_ascii=False)}
