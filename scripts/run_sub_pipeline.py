from __future__ import annotations

import argparse
import dataclasses as dc
import json
import pathlib as pl
import pickle
import sys

sys.path.insert(0, "src")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-pickle", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = pickle.loads(pl.Path(args.config_pickle).read_bytes())
    images = [f.read_bytes() for f in sorted(pl.Path(args.images_dir).glob("*.jpeg"), key=lambda p: int(p.stem))]

    from mausoleo.ocr.pipeline import run_pipeline

    issue = run_pipeline(config, images, date=args.date)
    out_path = pl.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dc.asdict(issue), indent=2, ensure_ascii=False))
    print(f"sub-pipeline {config.name} -> {out_path}")


if __name__ == "__main__":
    main()
