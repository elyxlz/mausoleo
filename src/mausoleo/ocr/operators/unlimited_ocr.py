from __future__ import annotations

import base64
import dataclasses as dc
import json
import pathlib as pl
import re
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


@register_operator(UnlimitedOcr, operation=OperatorType.MAP_BATCHES)
class UnlimitedOcrOperator(StatefulOperator[UnlimitedOcr]):
    def __init__(self, config: UnlimitedOcr) -> None:
        self.config = config
        if config.mock:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
        self.hf_model = (
            AutoModel.from_pretrained(config.model, trust_remote_code=True, torch_dtype=torch.bfloat16, use_safetensors=True).eval().cuda()
        )

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
            returned = self.hf_model.infer_multi(
                self.tokenizer,
                prompt=self.config.prompt,
                image_files=image_files,
                output_path=str(out_dir),
                image_size=self.config.image_size,
                max_length=self.config.max_length,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                ngram_window=self.config.ngram_window,
                save_results=True,
            )
            output_text = returned if isinstance(returned, str) else _read_saved_output(out_dir)

        page_texts = split_pages(output_text, len(raw_images))
        result = dict(batch)
        result["page_texts"] = [json.dumps(page_texts)]
        return result


def _read_saved_output(out_dir: pl.Path) -> str:
    candidates = sorted(out_dir.rglob("*.md")) + sorted(out_dir.rglob("*.mmd")) + sorted(out_dir.rglob("*.txt"))
    return "\n\n".join(f.read_text() for f in candidates)
