from __future__ import annotations

import dataclasses as dc
import json
import pathlib as pl

import pytest

from mausoleo.eval.evaluate import evaluate_issue
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.models import Issue
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr, YoloCrop
from mausoleo.ocr.pipeline import run_pipeline


FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100

GROUND_TRUTH_DIR = pl.Path(__file__).parent.parent / "eval" / "ground_truth"

ALL_CONFIGS = [
    OcrPipelineConfig(
        name="vlm_structured",
        operators=[VlmOcr(mock=True), MergePages(), ParseIssue()],
    ),
    OcrPipelineConfig(
        name="yolo_vlm_structured",
        operators=[YoloCrop(mock=True), VlmOcr(mock=True), MergePages(), ParseIssue()],
    ),
    OcrPipelineConfig(
        name="yolo_vlm_titles",
        operators=[YoloCrop(mock=True, separate_title_regions=True), VlmOcr(mock=True), MergePages(title_class_headlines=True), ParseIssue()],
    ),
]

EVAL_ISSUES = [
    ("1885-06-15", 4),
    ("1910-06-15", 6),
]


@pytest.mark.parametrize("config", ALL_CONFIGS, ids=lambda c: c.name)
@pytest.mark.parametrize("date,page_count", EVAL_ISSUES, ids=[d for d, _ in EVAL_ISSUES])
def test_pipeline_produces_valid_issue(config: OcrPipelineConfig, date: str, page_count: int) -> None:
    images = [FAKE_JPEG] * page_count
    issue = run_pipeline(config, images, date=date)

    assert isinstance(issue, Issue)
    assert issue.date == date
    assert issue.source == "il_messaggero"
    assert issue.page_count == page_count
    assert len(issue.articles) > 0

    for article in issue.articles:
        assert article.id.startswith(date)
        assert article.unit_type in {"article", "advertisement", "obituary", "notice", "editorial", "other"}
        assert len(article.paragraphs) >= 1
        for para in article.paragraphs:
            assert para.id.startswith(date)
            assert len(para.text) > 0


@pytest.mark.parametrize("config", ALL_CONFIGS, ids=lambda c: c.name)
def test_pipeline_eval_against_ground_truth(config: OcrPipelineConfig) -> None:
    for date, page_count in EVAL_ISSUES:
        gt_path = GROUND_TRUTH_DIR / date / "ground_truth.json"
        if not gt_path.exists():
            pytest.skip(f"no ground truth for {date}")

        images = [FAKE_JPEG] * page_count
        predicted = run_pipeline(config, images, date=date)
        expected = json.loads(gt_path.read_text())

        result = evaluate_issue(expected, dc.asdict(predicted), date=date)
        assert result.date == date
        assert result.total_pred_articles > 0
        assert result.total_gt_articles > 0
