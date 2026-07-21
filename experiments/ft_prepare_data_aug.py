from __future__ import annotations

import argparse
import json
import pathlib as pl
import random
import typing as tp

import numpy as np

REPO = pl.Path(__file__).resolve().parents[1]
PADDLEFT_DIR = REPO / "eval" / "autoresearch" / "paddleft"
SYNTH_JSONL = REPO / "eval" / "autoresearch" / "synth" / "synth_pairs.jsonl"
DATES = ["1885-06-15", "1895-06-15", "1910-06-15", "1925-06-15", "1935-06-15", "1952-06-15"]

Record = dict[str, tp.Any]


def read_jsonl(path: pl.Path) -> list[Record]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_images(records: list[Record]) -> None:
    missing = [r["images"][0] for r in records if not pl.Path(r["images"][0]).exists()]
    assert not missing, f"missing images: {missing[:3]} (+{len(missing) - 3 if len(missing) > 3 else 0})"


def real_repeat_count(n_real: int, n_synth: int, real_frac: float) -> int:
    target_real = real_frac / (1 - real_frac) * n_synth
    return max(1, round(target_real / n_real))


def mix_records(real: list[Record], synth: list[Record], repeat: int, rng: random.Random) -> list[Record]:
    mixed = real * repeat + synth
    rng.shuffle(mixed)
    return mixed


def label_chars(records: list[Record]) -> list[int]:
    return [len(r["messages"][1]["content"]) for r in records]


def print_stats(real: list[Record], synth: list[Record], repeat: int, mixed: list[Record]) -> None:
    n_real_eff = len(real) * repeat
    frac = n_real_eff / len(mixed)
    print(f"real: {len(real)} unique x{repeat} = {n_real_eff} effective")
    print(f"synth: {len(synth)}")
    print(f"mixed: {len(mixed)} (real fraction {frac:.2f})")
    for name, recs in (("real", real), ("synth", synth)):
        chars = label_chars(recs)
        print(f"{name} gt chars: mean {np.mean(chars):.0f}, p50 {np.percentile(chars, 50):.0f}, p95 {np.percentile(chars, 95):.0f}, max {max(chars)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("date", choices=DATES)
    parser.add_argument("--synth-count", type=int, default=2000)
    parser.add_argument("--real-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synth-jsonl", type=pl.Path, default=SYNTH_JSONL)
    args = parser.parse_args()
    tag = f"no{args.date}"
    real = read_jsonl(PADDLEFT_DIR / f"train_{tag}.jsonl")
    synth_all = read_jsonl(args.synth_jsonl)
    rng = random.Random(args.seed)
    synth = rng.sample(synth_all, min(args.synth_count, len(synth_all)))
    validate_images(synth)
    repeat = real_repeat_count(len(real), len(synth), args.real_frac)
    mixed = mix_records(real, synth, repeat, rng)
    out_path = PADDLEFT_DIR / f"train_aug_{tag}.jsonl"
    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in mixed) + "\n")
    print_stats(real, synth, repeat, mixed)
    print(f"wrote {out_path}")
    print(f"val unchanged (real-only): {PADDLEFT_DIR / f'val_{tag}.jsonl'}")


if __name__ == "__main__":
    main()
