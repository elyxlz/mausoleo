from __future__ import annotations

import dataclasses as dc
import json
import pathlib as pl
import typing as tp

from mausoleo.ocr.operators.base import BaseOperatorConfig, OperatorType, StatefulOperator, register_operator


@dc.dataclass(frozen=True, kw_only=True)
class SubPipelineOcr(BaseOperatorConfig):
    name: str = ""
    detector: tp.Literal["column_split", "yolo", "fullpage"] = "column_split"
    num_columns: int = 3
    overlap_pct: float = 0.03
    header_crop_pct: float = 0.03
    footer_crop_pct: float = 0.02
    yolo_model: str = "juliozhao/DocLayout-YOLO-DocStructBench"
    yolo_conf: float = 0.2
    yolo_min_area: int = 3000
    yolo_padding: int = 10
    yolo_merge_vertical_gap: int = 50
    yolo_merge_horizontal_overlap: float = 0.5
    model: str = ""
    prompt: str = ""
    max_tokens: int = 8192
    max_model_len: int = 12288
    backend: tp.Literal["vllm", "transformers"] = "vllm"
    load_in_4bit: bool = False
    temperature: float = 0.0
    cache_dir: str = "eval/predictions"
    gpu_fraction: float = 1.0


@register_operator(SubPipelineOcr, operation=OperatorType.MAP_BATCHES)
class SubPipelineOcrOperator(StatefulOperator[SubPipelineOcr]):
    def __init__(self, config: SubPipelineOcr) -> None:
        self.config = config
        self._vlm: tp.Any = None
        self._yolo: tp.Any = None
        cache_path = pl.Path(config.cache_dir)
        if not cache_path.is_absolute():
            cache_path = pl.Path(__file__).resolve().parent.parent.parent.parent.parent / config.cache_dir
        self._cache_dir = cache_path
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_vlm(self) -> tp.Any:
        if self._vlm is None:
            from mausoleo.ocr.operators.vlm_ocr import VlmOcr, VlmOcrOperator
            vlm_config = VlmOcr(
                model=self.config.model,
                prompt=self.config.prompt,
                max_tokens=self.config.max_tokens,
                max_model_len=self.config.max_model_len,
                backend=self.config.backend,
                load_in_4bit=self.config.load_in_4bit,
                temperature=self.config.temperature,
                gpu_fraction=self.config.gpu_fraction,
                mock=self.config.mock,
            )
            self._vlm = VlmOcrOperator(vlm_config)
        return self._vlm

    def _ensure_yolo(self) -> tp.Any:
        if self._yolo is None:
            from mausoleo.ocr.operators.yolo_crop import YoloCrop, YoloCropOperator
            yolo_config = YoloCrop(
                model=self.config.yolo_model,
                conf_threshold=self.config.yolo_conf,
                min_region_area=self.config.yolo_min_area,
                padding=self.config.yolo_padding,
                merge_vertical_gap=self.config.yolo_merge_vertical_gap,
                merge_horizontal_overlap=self.config.yolo_merge_horizontal_overlap,
                gpu_fraction=0.3,
                mock=self.config.mock,
            )
            self._yolo = YoloCropOperator(yolo_config)
        return self._yolo

    def _cache_path(self, date: str) -> pl.Path:
        return self._cache_dir / f"{self.config.name}_{date}.json"

    @staticmethod
    def _unwrap_batch_to_row(batch_result: dict[str, tp.Any]) -> dict[str, tp.Any]:
        out: dict[str, tp.Any] = {}
        for k, v in batch_result.items():
            if hasattr(v, "tolist"):
                v = v.tolist()
            if isinstance(v, list) and len(v) >= 1:
                v = v[0]
            out[k] = v
        return out

    def _run_detector(self, row: dict[str, tp.Any]) -> dict[str, tp.Any]:
        if self.config.detector == "column_split":
            from mausoleo.ocr.operators.column_split import ColumnSplit, column_split
            cs_config = ColumnSplit(
                num_columns=self.config.num_columns,
                overlap_pct=self.config.overlap_pct,
                header_crop_pct=self.config.header_crop_pct,
                footer_crop_pct=self.config.footer_crop_pct,
                mock=self.config.mock,
            )
            return column_split(row, config=cs_config)
        if self.config.detector == "yolo":
            yolo = self._ensure_yolo()
            batch = {k: [v] for k, v in row.items()}
            result = yolo(batch)
            return self._unwrap_batch_to_row(result)
        return row

    def _run_vlm(self, row: dict[str, tp.Any]) -> dict[str, tp.Any]:
        vlm = self._ensure_vlm()
        batch = {k: [v] for k, v in row.items()}
        result = vlm(batch)
        return self._unwrap_batch_to_row(result)

    def _run_merge_parse(self, row: dict[str, tp.Any]) -> str:
        from mausoleo.ocr.operators.merge import MergePages, merge_pages
        from mausoleo.ocr.operators.parse import ParseIssue, parse_issue
        row = merge_pages(row, config=MergePages(mock=self.config.mock))
        row = parse_issue(row, config=ParseIssue(mock=self.config.mock))
        return str(row["issue_json"])

    def __call__(self, batch: dict[str, tp.Any]) -> dict[str, tp.Any]:
        row: dict[str, tp.Any] = {}
        for k, v in batch.items():
            if hasattr(v, "tolist"):
                v = v.tolist()
            if isinstance(v, list) and len(v) >= 1:
                v = v[0]
            row[k] = v
        date = str(row.get("date", ""))
        cache_path = self._cache_path(date)

        if cache_path.exists():
            issue_json = cache_path.read_text()
        else:
            row = self._run_detector(row)
            row = self._run_vlm(row)
            issue_json = self._run_merge_parse(row)
            cache_path.write_text(issue_json)

        result = dict(batch)
        result[self.config.name] = [issue_json]
        return result
