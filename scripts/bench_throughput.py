from __future__ import annotations

import inspect
import json
import pathlib as pl
import sys
import time
import typing as tp

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from mausoleo.ocr.models import issue_from_dict
from mausoleo.ocr.operators.base import OPERATOR_REGISTRY, BaseOperatorConfig, OperatorType
from mausoleo.ocr.pipeline import encode_images

GROUND_TRUTH_DIR = pl.Path("eval/ground_truth")
DEFAULT_DATES = ["1885-06-15", "1910-06-15", "1943-07-03", "1943-07-15", "1943-07-25"]
CORPUS_PAGES = 175_000
GPUS = 2


def load_images(date: str) -> list[bytes]:
    issue_dir = GROUND_TRUTH_DIR / date
    return [f.read_bytes() for f in sorted(issue_dir.glob("*.jpeg"), key=lambda p: int(p.stem))]


def row_to_batch(row: dict[str, tp.Any]) -> dict[str, tp.Any]:
    return {k: [v] for k, v in row.items()}


def batch_to_row(batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
    return {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in batch.items()}


def build_stages(operators: tp.Sequence[BaseOperatorConfig]) -> tuple[list[tuple[BaseOperatorConfig, tp.Any, OperatorType]], float]:
    stages: list[tuple[BaseOperatorConfig, tp.Any, OperatorType]] = []
    load_s = 0.0
    for op_config in operators:
        entry = OPERATOR_REGISTRY[type(op_config)]
        if inspect.isclass(entry.impl):
            t0 = time.time()
            impl = entry.impl(op_config)
            load_s += time.time() - t0
        else:
            impl = entry.impl
        stages.append((op_config, impl, entry.operation))
    return stages, load_s


def run_issue(stages: list[tuple[BaseOperatorConfig, tp.Any, OperatorType]], date: str, images: list[bytes]) -> tuple[float, int]:
    row: dict[str, tp.Any] = {
        "issue_id": date,
        "date": date,
        "source": "il_messaggero",
        "page_count": len(images),
        "images_b64": encode_images(images),
    }
    t0 = time.time()
    for op_config, impl, operation in stages:
        if operation == OperatorType.MAP_BATCHES:
            row = batch_to_row(impl(row_to_batch(row)))
        elif operation == OperatorType.MAP:
            row = impl(row, config=op_config)
        else:
            raise ValueError(f"bench does not support {operation}")
    issue = issue_from_dict(json.loads(row["issue_json"]))
    return time.time() - t0, len(issue.articles)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: bench_throughput.py <config> [<date> ...]")
        raise SystemExit(1)
    config_name = sys.argv[1]
    dates = sys.argv[2:] or DEFAULT_DATES

    from run_real_ocr import load_config

    config = load_config(config_name)
    stages, load_s = build_stages(config.operators)
    print(f"model/operator load: {load_s:.1f}s (amortized to ~0 at corpus scale)")

    per_issue: list[tuple[str, float, int, int]] = []
    for date in dates:
        images = load_images(date)
        if not images:
            print(f"{date}: no images, skipping")
            continue
        elapsed, n_articles = run_issue(stages, date, images)
        per_issue.append((date, elapsed, len(images), n_articles))
        print(f"{date}: {elapsed:.1f}s / {len(images)}pp = {elapsed / len(images):.2f} GPU-s/page ({n_articles} articles)")

    if len(per_issue) > 1:
        steady = per_issue[1:]
        total_s = sum(e for _, e, _, _ in steady)
        total_pp = sum(p for _, _, p, _ in steady)
        gpu_s_page = total_s / total_pp
        days = CORPUS_PAGES * gpu_s_page / GPUS / 86400
        print(f"\nsteady-state (excl. first issue warmup): {gpu_s_page:.2f} GPU-s/page over {total_pp}pp")
        print(f"corpus extrapolation: {CORPUS_PAGES} pages on {GPUS} GPUs = {days:.1f} days (budget: 7-14)")


if __name__ == "__main__":
    main()
