from __future__ import annotations

import dataclasses as dc

from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.models import Issue, extract_full_text, issue_from_dict
from mausoleo.ocr.operators import LlmCleanup, MergePages, ParseIssue, VlmOcr, WholeIssueVlm
from mausoleo.ocr.pipeline import run_pipeline


FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100


def test_mock_vlm_pipeline() -> None:
    config = OcrPipelineConfig(
        name="mock_vlm",
        operators=[VlmOcr(mock=True), LlmCleanup(mock=True), ParseIssue()],
    )
    issue = run_pipeline(config, [FAKE_JPEG] * 4, date="1885-06-15")

    assert isinstance(issue, Issue)
    assert issue.date == "1885-06-15"
    assert issue.page_count == 4
    assert len(issue.articles) == 4

    for i, article in enumerate(issue.articles):
        assert article.id == f"1885-06-15_a{i:02d}"
        assert len(article.paragraphs) >= 1


def test_mock_whole_issue_pipeline() -> None:
    config = OcrPipelineConfig(
        name="mock_whole_issue",
        operators=[WholeIssueVlm(mock=True), ParseIssue()],
    )
    issue = run_pipeline(config, [FAKE_JPEG] * 6, date="1940-04-01")

    assert isinstance(issue, Issue)
    assert issue.page_count == 6
    assert len(issue.articles) == 6


def test_issue_serialization() -> None:
    config = OcrPipelineConfig(
        name="mock",
        operators=[VlmOcr(mock=True), LlmCleanup(mock=True), ParseIssue()],
    )
    issue = run_pipeline(config, [FAKE_JPEG] * 2, date="1910-06-15")

    roundtripped = issue_from_dict(dc.asdict(issue))
    assert roundtripped.date == issue.date
    assert len(roundtripped.articles) == len(issue.articles)
    assert roundtripped.articles[0].paragraphs[0].text == issue.articles[0].paragraphs[0].text


def test_extract_full_text() -> None:
    config = OcrPipelineConfig(
        name="mock",
        operators=[VlmOcr(mock=True), LlmCleanup(mock=True), ParseIssue()],
    )
    issue = run_pipeline(config, [FAKE_JPEG] * 2, date="1910-06-15")

    text = extract_full_text(issue)
    assert len(text) > 0
    assert "Articolo" in text
