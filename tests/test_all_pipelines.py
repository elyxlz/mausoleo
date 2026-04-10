from __future__ import annotations

import pathlib as pl

import pytest

from mausoleo.eval.evaluate import evaluate_issue, load_ground_truth
from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.models import Issue
from mausoleo.ocr.operators import LlmCleanup, MergePages, ParseIssue, Preprocess, SuryaOcr, VlmOcr, WholeIssueVlm, YoloLayout
from mausoleo.ocr.pipeline import run_pipeline


FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100

GROUND_TRUTH_DIR = pl.Path(__file__).parent.parent / "eval" / "ground_truth"

ALL_CONFIGS = [
    OcrPipelineConfig(
        name="qwen_vl_7b_structured",
        operators=[VlmOcr(mock=True), MergePages(), ParseIssue()],
    ),
    OcrPipelineConfig(
        name="qwen_vl_7b_raw_cleanup",
        operators=[VlmOcr(mock=True), LlmCleanup(mock=True), ParseIssue()],
    ),
    OcrPipelineConfig(
        name="qwen_vl_72b_raw_cleanup",
        operators=[VlmOcr(mock=True), LlmCleanup(mock=True), ParseIssue()],
    ),
    OcrPipelineConfig(
        name="qwen_vl_7b_whole_issue",
        operators=[WholeIssueVlm(mock=True), ParseIssue()],
    ),
    OcrPipelineConfig(
        name="internvl_8b_raw_cleanup",
        operators=[VlmOcr(mock=True), LlmCleanup(mock=True), ParseIssue()],
    ),
    OcrPipelineConfig(
        name="got_ocr2_cleanup",
        operators=[VlmOcr(mock=True), LlmCleanup(mock=True), ParseIssue()],
    ),
    OcrPipelineConfig(
        name="surya_cleanup",
        operators=[SuryaOcr(mock=True), LlmCleanup(mock=True), ParseIssue()],
    ),
    OcrPipelineConfig(
        name="preprocess_qwen_cleanup",
        operators=[Preprocess(mock=True), VlmOcr(mock=True), LlmCleanup(mock=True), ParseIssue()],
    ),
    OcrPipelineConfig(
        name="yolo_qwen_structured",
        operators=[YoloLayout(mock=True), VlmOcr(mock=True), MergePages(), ParseIssue()],
    ),
    OcrPipelineConfig(
        name="yolo_surya_cleanup",
        operators=[YoloLayout(mock=True), SuryaOcr(mock=True), LlmCleanup(mock=True), ParseIssue()],
    ),
]

EVAL_ISSUES = [
    ("1885-06-15", 4),
    ("1910-06-15", 6),
    ("1940-04-01", 6),
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
        expected = load_ground_truth(gt_path)

        result = evaluate_issue(predicted, expected)
        assert result.issue_id == date
        assert result.predicted_articles > 0
        assert result.expected_articles > 0
