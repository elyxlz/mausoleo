from __future__ import annotations

import json
import pathlib as pl

import pytest

from mausoleo.eval.evaluate import evaluate_issue, load_images
from mausoleo.eval.metrics import compute_cer, compute_kendalls_tau, compute_wer
from mausoleo.ocr.models import Article, Issue, Paragraph


def test_cer_identical() -> None:
    assert compute_cer("hello world", "hello world") == 0.0


def test_cer_different() -> None:
    cer = compute_cer("hello", "hallo")
    assert 0.0 < cer < 1.0


def test_wer_identical() -> None:
    assert compute_wer("hello world", "hello world") == 0.0


def test_wer_different() -> None:
    wer = compute_wer("hello world", "hallo world")
    assert 0.0 < wer <= 1.0


def test_kendalls_tau_identical() -> None:
    assert compute_kendalls_tau([1, 2, 3], [1, 2, 3]) == 1.0


def test_kendalls_tau_reversed() -> None:
    assert compute_kendalls_tau([3, 2, 1], [1, 2, 3]) == -1.0


def test_evaluate_issue() -> None:
    predicted = Issue(
        date="1885-06-15",
        source="il_messaggero",
        page_count=4,
        articles=[
            Article(
                id="1885-06-15_a00",
                unit_type="article",
                headline="Test",
                paragraphs=[Paragraph(id="1885-06-15_a00_p00", text="Testo di prova")],
                page_span=[1],
                position_in_issue=0,
            ),
        ],
    )
    expected = Issue(
        date="1885-06-15",
        source="il_messaggero",
        page_count=4,
        articles=[
            Article(
                id="1885-06-15_a00",
                unit_type="article",
                headline="Test",
                paragraphs=[Paragraph(id="1885-06-15_a00_p00", text="Testo di prova")],
                page_span=[1],
                position_in_issue=0,
            ),
        ],
    )

    result = evaluate_issue(predicted, expected)
    assert result.cer == 0.0
    assert result.wer == 0.0
    assert result.predicted_articles == 1
    assert result.expected_articles == 1


def test_load_images(tmp_path: pl.Path) -> None:
    for i in range(3):
        (tmp_path / f"{i + 1}.jpeg").write_bytes(b"\xff\xd8" + b"\x00" * 50)

    images = load_images(tmp_path)
    assert len(images) == 3
    assert all(isinstance(img, bytes) for img in images)
