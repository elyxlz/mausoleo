from __future__ import annotations

import glob
import pathlib as pl
import shutil
import sys

from safetensors.torch import load_file, save_file

PADDLEFT = pl.Path("eval/autoresearch/paddleft")
CODE_FILES = ["configuration_paddleocr_vl.py", "modeling_paddleocr_vl.py",
              "image_processing_paddleocr_vl.py", "processing_paddleocr_vl.py",
              "added_tokens.json", "special_tokens_map.json", "tokenizer.model", "inference.yml"]


def _base_snapshot() -> pl.Path:
    hits = glob.glob(str(pl.Path.home() / ".cache/huggingface/hub/models--PaddlePaddle--PaddleOCR-VL-1.6/snapshots/*/"))
    if not hits:
        raise RuntimeError("base PaddleOCR-VL-1.6 snapshot not found in HF cache")
    return pl.Path(hits[0])


def _remap(k: str) -> str:
    return "visual." + k[len("model.visual."):] if k.startswith("model.visual.") else k


def fix_merged(date: str) -> None:
    base = _base_snapshot()
    merged = PADDLEFT / f"merged_no{date}"
    if not merged.is_dir():
        raise RuntimeError(f"missing {merged}")
    for f in CODE_FILES:
        if not (merged / f).exists() and (base / f).exists():
            shutil.copy(base / f, merged / f)
    shutil.copy(base / "config.json", merged / "config.json")
    m = load_file(str(merged / "model.safetensors"))
    b = load_file(str(base / "model.safetensors"))
    out = {_remap(k): v for k, v in m.items()}
    for k, v in b.items():
        out.setdefault(k, v)
    assert set(out.keys()) == set(b.keys()), f"key mismatch: {set(b) - set(out)}"
    save_file(out, str(merged / "model.safetensors"), metadata={"format": "pt"})
    print(f"fixed {merged}: {len(m)} -> {len(out)} keys (vllm convention)")


if __name__ == "__main__":
    fix_merged(sys.argv[1])
