from __future__ import annotations

import dataclasses as dc
import json
import os
import pathlib as pl
import subprocess
import sys
import typing as tp

from mausoleo.ocr.operators.base import BaseOperatorConfig, OperatorType, register_operator


@dc.dataclass(frozen=True, kw_only=True)
class ParallelEnsembleOcr(BaseOperatorConfig):
    gpu0_chain: tuple[str, ...] = ()
    gpu1_chain: tuple[str, ...] = ()
    primary_name: str = ""
    replacement_chain: tuple[tuple[str, float, float], ...] = ()
    additive_sources: tuple[tuple[str, float, float], ...] = ()
    quality_select_sources: tuple[str, ...] = ()
    crosspage_col1_sources: tuple[str, ...] = ()
    min_quality_delta: float = 0.10
    headline_delta: float = 0.15
    cache_dir: str = "eval/predictions"


def _repo_root() -> pl.Path:
    return pl.Path(__file__).resolve().parent.parent.parent.parent.parent


def _run_chain(configs: list[str], date: str, gpu_index: int, log_path: pl.Path) -> int:
    if not configs:
        return 0
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["RAY_ADDRESS"] = ""
    root = _repo_root()
    cmd = [sys.executable, str(root / "scripts" / "run_real_ocr.py"), *configs, date]
    with open(log_path, "w") as f:
        return subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=str(root)).wait()


def _launch_parallel(configs0: list[str], configs1: list[str], date: str, log_dir: pl.Path) -> tuple[int, int]:
    if not configs0 and not configs1:
        return 0, 0
    log_dir.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[int, subprocess.Popen[bytes]]] = []
    root = _repo_root()
    for gpu, chain in ((0, configs0), (1, configs1)):
        if not chain:
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["RAY_ADDRESS"] = ""
        cmd = [sys.executable, str(root / "scripts" / "run_real_ocr.py"), *chain, date]
        log = open(log_dir / f"parallel_ensemble_gpu{gpu}_{date}.log", "wb")
        procs.append((gpu, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(root))))
    rc0 = rc1 = 0
    for gpu, proc in procs:
        rc = proc.wait()
        if gpu == 0:
            rc0 = rc
        else:
            rc1 = rc
    return rc0, rc1


@register_operator(ParallelEnsembleOcr, operation=OperatorType.MAP)
def _parallel_ensemble_ocr(row: dict[str, tp.Any], *, config: ParallelEnsembleOcr) -> dict[str, tp.Any]:
    from mausoleo.ocr.merge import (
        merge_with_replacement,
        replace_with_pairs,
        select_best_text,
        trim_predictions,
    )

    date = str(row.get("date", ""))
    cache = pl.Path(config.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    missing0 = [name for name in config.gpu0_chain if not (cache / f"{name}_{date}.json").exists()]
    missing1 = [name for name in config.gpu1_chain if not (cache / f"{name}_{date}.json").exists()]
    if missing0 or missing1:
        _launch_parallel(missing0, missing1, date, _repo_root() / "logs")

    def load(name: str) -> dict[str, tp.Any]:
        return trim_predictions(json.loads((cache / f"{name}_{date}.json").read_text()))

    current = load(config.primary_name)
    for src, ov, rt in config.replacement_chain:
        if not (cache / f"{src}_{date}.json").exists():
            continue
        current = merge_with_replacement(current, load(src), overlap_threshold=ov, replace_ratio=rt)
    for src, ov, rt in config.additive_sources:
        if not (cache / f"{src}_{date}.json").exists():
            continue
        current = merge_with_replacement(current, load(src), overlap_threshold=ov, replace_ratio=rt)

    qs_list: list[dict[str, tp.Any]] = []
    for name in config.quality_select_sources:
        p = cache / f"{name}_{date}.json"
        if p.exists():
            qs_list.append(json.loads(p.read_text()))
    current = select_best_text(current, qs_list, min_quality_delta=config.min_quality_delta, headline_delta=config.headline_delta)
    current = trim_predictions(current)

    if config.crosspage_col1_sources:
        col1_predictions: list[dict[str, tp.Any]] = []
        for name in config.crosspage_col1_sources:
            p = cache / f"{name}_{date}.json"
            if p.exists():
                col1_predictions.append(json.loads(p.read_text()))
        if col1_predictions:
            current, _, _ = replace_with_pairs(current, col1_predictions)

    return {**row, "result_json": json.dumps(current, ensure_ascii=False)}
