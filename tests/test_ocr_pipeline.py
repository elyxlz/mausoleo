from __future__ import annotations

import dataclasses as dc

from mausoleo.ocr.config import OcrPipelineConfig
from mausoleo.ocr.models import Issue, extract_full_text, issue_from_dict
from mausoleo.ocr.operators import MergePages, ParseIssue, VlmOcr
from mausoleo.ocr.pipeline import run_pipeline


FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100


def test_mock_vlm_pipeline() -> None:
    config = OcrPipelineConfig(
        name="mock_vlm",
        operators=[VlmOcr(mock=True), MergePages(), ParseIssue()],
    )
    issue = run_pipeline(config, [FAKE_JPEG] * 4, date="1885-06-15")

    assert isinstance(issue, Issue)
    assert issue.date == "1885-06-15"
    assert issue.page_count == 4
    assert len(issue.articles) == 4

    for i, article in enumerate(issue.articles):
        assert article.id == f"1885-06-15_a{i:02d}"
        assert len(article.paragraphs) >= 1


def test_issue_serialization() -> None:
    config = OcrPipelineConfig(
        name="mock",
        operators=[VlmOcr(mock=True), MergePages(), ParseIssue()],
    )
    issue = run_pipeline(config, [FAKE_JPEG] * 2, date="1910-06-15")

    roundtripped = issue_from_dict(dc.asdict(issue))
    assert roundtripped.date == issue.date
    assert len(roundtripped.articles) == len(issue.articles)
    assert roundtripped.articles[0].paragraphs[0].text == issue.articles[0].paragraphs[0].text


def test_extract_full_text() -> None:
    config = OcrPipelineConfig(
        name="mock",
        operators=[VlmOcr(mock=True), MergePages(), ParseIssue()],
    )
    issue = run_pipeline(config, [FAKE_JPEG] * 2, date="1910-06-15")

    text = extract_full_text(issue)
    assert len(text) > 0
    assert "Mock OCR output" in text


def test_split_markdown_articles() -> None:
    from mausoleo.ocr.operators.merge_markdown import split_markdown_articles

    markdown = "# TITOLO UNO\n\nPrimo paragrafo.\n\nSecondo paragrafo.\n\n## Titolo due\n\nAltro testo."
    articles = split_markdown_articles(markdown)

    assert len(articles) == 2
    assert articles[0]["headline"] == "TITOLO UNO"
    assert len(articles[0]["paragraphs"]) == 2
    assert articles[1]["headline"] == "Titolo due"
    assert articles[1]["paragraphs"][0]["text"] == "Altro testo."


def test_split_markdown_articles_no_headings() -> None:
    from mausoleo.ocr.operators.merge_markdown import split_markdown_articles

    articles = split_markdown_articles("solo testo\n\nsenza titoli")

    assert len(articles) == 1
    assert articles[0]["headline"] is None
    assert len(articles[0]["paragraphs"]) == 2


def test_split_markdown_articles_bold_headline_and_noise() -> None:
    from mausoleo.ocr.operators.merge_markdown import split_markdown_articles

    markdown = "**UN FURTO**\n\n![figura](img.png) Testo con **enfasi** rimossa."
    articles = split_markdown_articles(markdown)

    assert articles[0]["headline"] == "UN FURTO"
    assert articles[0]["paragraphs"][0]["text"] == "Testo con enfasi rimossa."
