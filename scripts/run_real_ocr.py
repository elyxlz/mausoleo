from __future__ import annotations

import dataclasses as dc
import importlib.util
import json
import pathlib as pl
import signal
import sys
import time
import traceback

import ray

from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.models import extract_full_text
from mausoleo.ocr.pipeline import run_pipeline


class PipelineTimeoutError(Exception):
    pass


def _timeout_handler(signum: int, frame: object) -> None:
    raise PipelineTimeoutError("timed out")


def _shutdown_ray() -> None:
    try:
        if ray.is_initialized():
            ray.shutdown()
            print("ray shutdown complete")
    except Exception:
        pass


GROUND_TRUTH_DIR = pl.Path("eval/ground_truth")
PREDICTIONS_DIR = pl.Path("eval/predictions")
CONFIGS_DIR = pl.Path("configs/ocr")
ISSUE_DATES = ["1885-06-15", "1910-06-15", "1940-04-01"]


def load_config(name: str) -> OcrPipelineConfig:
    config_path = CONFIGS_DIR / f"{name}.py"
    if not config_path.exists():
        raise FileNotFoundError(f"no config at {config_path}")
    spec = importlib.util.spec_from_file_location(f"configs.ocr.{name}", config_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.config


def list_configs() -> list[str]:
    return sorted(p.stem for p in CONFIGS_DIR.glob("*.py"))


def load_issue_images(issue_dir: pl.Path) -> list[bytes]:
    return [f.read_bytes() for f in sorted(issue_dir.glob("*.jpeg"), key=lambda p: int(p.stem))]


def run_single(config_name: str, issue_date: str, force: bool = False) -> bool:
    output_path = PREDICTIONS_DIR / f"{config_name}_{issue_date}.json"
    if output_path.exists() and not force:
        print(f"SKIP {config_name} {issue_date} (exists)")
        return True

    issue_dir = GROUND_TRUTH_DIR / issue_date
    if not issue_dir.exists():
        print(f"no images at {issue_dir}")
        return False

    try:
        config = load_config(config_name)
    except Exception as e:
        print(f"FAILED loading config {config_name}: {e}")
        return False

    images = load_issue_images(issue_dir)
    print(f"running {config_name} on {issue_date} ({len(images)} pages)", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    signal.signal(signal.SIGALRM, _timeout_handler)
    timeout_secs = 5400
    signal.alarm(timeout_secs)
    try:
        issue = run_pipeline(config, images, date=issue_date)
    except PipelineTimeoutError:
        print(f"FAILED: timed out after {timeout_secs}s")
        _shutdown_ray()
        return False
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        _shutdown_ray()
        return False
    finally:
        signal.alarm(0)

    elapsed = time.time() - t0
    n_articles = len(issue.articles)
    total_chars = len(extract_full_text(issue))
    print(f"-> {n_articles} articles | {total_chars} chars | {elapsed:.1f}s")

    for i, art in enumerate(issue.articles):
        headline = art.headline or "(no headline)"
        chars = sum(len(p.text) for p in art.paragraphs)
        print(f"   [{i}] {art.unit_type}: {headline} ({chars}c, pages={art.page_span})")

    print(f"preview: {extract_full_text(issue)[:400]}...")
    output_path.write_text(json.dumps(dc.asdict(issue), indent=2, ensure_ascii=False))
    print(f"saved: {output_path}")
    return True


def main() -> None:
    import re

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        configs = list_configs()
        print(f"usage: run_real_ocr.py <config|all> [<config> ...] [date|all] [--force]")
        print(f"\navailable configs ({len(configs)}):")
        for c in configs:
            print(f"  {c}")
        return

    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    force = "--force" in sys.argv

    raw_configs: list[str] = []
    raw_dates: list[str] = []
    for tok in sys.argv[1:]:
        if tok == "--force":
            continue
        if date_re.match(tok) or tok == "all":
            raw_dates.append(tok)
        else:
            raw_configs.append(tok)

    if not raw_configs:
        raw_configs = ["all"]
    if not raw_dates:
        raw_dates = ["all"]

    dates: list[str] = []
    for d in raw_dates:
        if d == "all":
            dates.extend(ISSUE_DATES)
        else:
            dates.append(d)
    configs: list[str] = []
    for c in raw_configs:
        if c == "all":
            configs.extend(list_configs())
        else:
            configs.append(c)

    total = len(configs) * len(dates)
    print(f"running {len(configs)} configs x {len(dates)} dates = {total} runs\n")

    results: list[tuple[str, str, bool]] = []
    for cfg in configs:
        for d in dates:
            print(f"\n{'=' * 60}")
            ok = run_single(cfg, d, force=force)
            results.append((cfg, d, ok))

    ok_count = sum(1 for _, _, s in results if s)
    print(f"\n\n{'=' * 60}")
    print(f"DONE: {ok_count}/{total} succeeded")
    for cfg, d, s in results:
        print(f"  [{'OK' if s else 'FAIL'}] {cfg} | {d}")


if __name__ == "__main__":
    main()
