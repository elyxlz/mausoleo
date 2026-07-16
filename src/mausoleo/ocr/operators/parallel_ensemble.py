from __future__ import annotations

import base64
import dataclasses as dc
import json
import os
import pathlib as pl
import pickle
import queue
import subprocess
import sys
import tempfile
import threading
import typing as tp

from mausoleo.ocr.operators.base import BaseOperatorConfig, OperatorType, register_operator


@dc.dataclass(frozen=True, kw_only=True)
class ParallelEnsembleOcr(BaseOperatorConfig):
    sub_configs: tuple[tp.Any, ...] = ()
    primary_name: str = ""
    replacement_chain: tuple[tuple[str, float, float], ...] = ()
    additive_sources: tuple[tuple[str, float, float], ...] = ()
    quality_select_sources: tuple[str, ...] = ()
    crosspage_col1_sources: tuple[str, ...] = ()
    min_quality_delta: float = 0.10
    headline_delta: float = 0.15
    cache_dir: str = "eval/predictions"
    num_gpus: int = 2
    sub_timeout_s: int = 3600


def _repo_root() -> pl.Path:
    return pl.Path(__file__).resolve().parent.parent.parent.parent.parent


def _write_images(images_b64: str, target: pl.Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for i, b64 in enumerate(images_b64.split("|")):
        (target / f"{i + 1}.jpeg").write_bytes(base64.b64decode(b64))


def _run_sub_config(sub: tp.Any, gpu: int, images_dir: pl.Path, date: str, out_path: pl.Path, work_dir: pl.Path, timeout_s: int) -> None:
    root = _repo_root()
    cfg_pickle = work_dir / f"{sub.name}.pickle"
    cfg_pickle.write_bytes(pickle.dumps(sub))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["RAY_ADDRESS"] = ""
    cmd = [
        sys.executable,
        str(root / "scripts" / "run_sub_pipeline.py"),
        "--config-pickle",
        str(cfg_pickle),
        "--images-dir",
        str(images_dir),
        "--date",
        date,
        "--output",
        str(out_path),
    ]
    log_path = work_dir / f"{sub.name}_gpu{gpu}.log"
    with open(log_path, "wb") as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(root), timeout=timeout_s)
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"sub-pipeline {sub.name} failed on gpu{gpu}, log: {log_path}")


def _run_queue(
    subs: list[tp.Any], num_gpus: int, images_dir: pl.Path, date: str, cache: pl.Path, work_dir: pl.Path, timeout_s: int
) -> list[str]:
    task_queue: queue.Queue[tp.Any] = queue.Queue()
    for sub in subs:
        task_queue.put(sub)
    errors: list[str] = []

    def worker(gpu: int) -> None:
        while True:
            try:
                sub = task_queue.get_nowait()
            except queue.Empty:
                return
            out_path = cache / f"{sub.name}_{date}.json"
            try:
                _run_sub_config(sub, gpu, images_dir, date, out_path, work_dir, timeout_s)
            except Exception as exc:
                errors.append(f"{sub.name}: {exc}")

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in range(num_gpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


@register_operator(ParallelEnsembleOcr, operation=OperatorType.MAP)
def parallel_ensemble_ocr(row: dict[str, tp.Any], *, config: ParallelEnsembleOcr) -> dict[str, tp.Any]:
    from mausoleo.ocr.merge import (
        merge_with_replacement,
        replace_with_pairs,
        select_best_text,
        trim_predictions,
    )

    date = str(row.get("date", ""))
    cache = pl.Path(config.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    missing = [sub for sub in config.sub_configs if not (cache / f"{sub.name}_{date}.json").exists()]
    if missing:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = pl.Path(tmp)
            images_dir = work_dir / "images"
            _write_images(str(row["images_b64"]), images_dir)
            errors = _run_queue(missing, config.num_gpus, images_dir, date, cache, work_dir, config.sub_timeout_s)
            if errors:
                for log_file in sorted(work_dir.glob("*.log")):
                    persisted = _repo_root() / "logs" / f"ensemble_{date}_{log_file.name}"
                    persisted.parent.mkdir(parents=True, exist_ok=True)
                    persisted.write_bytes(log_file.read_bytes())
                raise RuntimeError(f"ensemble sub-pipelines failed: {errors}")

    def load(name: str) -> dict[str, tp.Any]:
        return trim_predictions(json.loads((cache / f"{name}_{date}.json").read_text()))

    available = {sub.name for sub in config.sub_configs if (cache / f"{sub.name}_{date}.json").exists()}
    current = load(config.primary_name)
    for src, ov, rt in [*config.replacement_chain, *config.additive_sources]:
        if src not in available:
            continue
        current = merge_with_replacement(current, load(src), overlap_threshold=ov, replace_ratio=rt)

    qs_list = [json.loads((cache / f"{name}_{date}.json").read_text()) for name in config.quality_select_sources if name in available]
    current = select_best_text(current, qs_list, min_quality_delta=config.min_quality_delta, headline_delta=config.headline_delta)
    current = trim_predictions(current)

    if config.crosspage_col1_sources:
        col1_predictions = [
            json.loads((cache / f"{name}_{date}.json").read_text()) for name in config.crosspage_col1_sources if name in available
        ]
        if col1_predictions:
            current, _, _ = replace_with_pairs(current, col1_predictions)

    return {**row, "result_json": json.dumps(current, ensure_ascii=False)}
