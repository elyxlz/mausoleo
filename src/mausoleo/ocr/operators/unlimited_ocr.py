from __future__ import annotations

import base64
import dataclasses as dc
import json
import pathlib as pl
import re
import subprocess
import tempfile
import typing as tp

from mausoleo.ocr.operators.base import BaseOperatorConfig, OperatorType, StatefulOperator, register_operator

_PAGE_SPLIT_RE = re.compile(r"<-{2,}\s*[Pp]age\s*[Ss]plit\s*-{2,}>|\n-{3,}\s*[Pp]age\s+\d+\s*-{3,}\n|<\|page_sep\|>")


def split_pages(output_text: str, expected_pages: int) -> list[str]:
    parts = [p.strip() for p in _PAGE_SPLIT_RE.split(output_text) if p.strip()]
    if len(parts) == expected_pages:
        return parts
    return [output_text.strip()] + [""] * (expected_pages - 1) if expected_pages > 0 else [output_text.strip()]


@dc.dataclass(frozen=True, kw_only=True)
class UnlimitedOcr(BaseOperatorConfig):
    model: str = "baidu/Unlimited-OCR"
    prompt: str = "<image>Multi page parsing."
    image_size: int = 1024
    max_length: int = 32768
    no_repeat_ngram_size: int = 35
    ngram_window: int = 1024
    python_bin: str = "~/unlimited_env/bin/python"
    runner_script: str = "scripts/run_unlimited_standalone.py"
    timeout_s: int = 3600


@register_operator(UnlimitedOcr, operation=OperatorType.MAP_BATCHES)
class UnlimitedOcrOperator(StatefulOperator[UnlimitedOcr]):
    def __init__(self, config: UnlimitedOcr) -> None:
        self.config = config

    def __call__(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        if self.config.mock:
            return self._mock_call(batch)
        return self._infer_call(batch)

    def _mock_call(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        images_b64 = str(batch["images_b64"][0])
        page_count = len(images_b64.split("|"))
        page_texts = [f"# Titolo pagina {i + 1}\n\nTesto simulato." for i in range(page_count)]
        result = dict(batch)
        result["page_texts"] = [json.dumps(page_texts)]
        return result

    def _infer_call(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        images_b64 = str(batch["images_b64"][0])
        raw_images = [base64.b64decode(b64) for b64 in images_b64.split("|")]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = pl.Path(tmp)
            image_files = []
            for i, img_bytes in enumerate(raw_images):
                img_path = tmp_dir / f"page_{i + 1:03d}.jpeg"
                img_path.write_bytes(img_bytes)
                image_files.append(str(img_path))
            out_dir = tmp_dir / "out"
            out_dir.mkdir()
            cmd = [
                str(pl.Path(self.config.python_bin).expanduser()),
                self.config.runner_script,
                "--model",
                self.config.model,
                "--prompt",
                self.config.prompt,
                "--output-dir",
                str(out_dir),
                "--image-size",
                str(self.config.image_size),
                "--max-length",
                str(self.config.max_length),
                "--no-repeat-ngram-size",
                str(self.config.no_repeat_ngram_size),
                "--ngram-window",
                str(self.config.ngram_window),
                *image_files,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.config.timeout_s)
            if proc.returncode != 0:
                raise RuntimeError(f"unlimited-ocr runner failed: {proc.stderr[-2000:]}")
            runner_result = json.loads((out_dir / "runner_result.json").read_text())
            returned_text = runner_result.get("returned_text")
            output_text = returned_text if isinstance(returned_text, str) and returned_text.strip() else _read_saved_output(out_dir)

        page_texts = split_pages(output_text, len(raw_images))
        result = dict(batch)
        result["page_texts"] = [json.dumps(page_texts)]
        return result


def _read_saved_output(out_dir: pl.Path) -> str:
    candidates = sorted(out_dir.rglob("*.md")) + sorted(out_dir.rglob("*.mmd")) + sorted(out_dir.rglob("*.txt"))
    return "\n\n".join(f.read_text() for f in candidates)
