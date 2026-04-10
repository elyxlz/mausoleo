from __future__ import annotations

import dataclasses as dc
import json
import pathlib as pl

import httpx

from mausoleo.eval.metrics import compute_cer, compute_wer
from mausoleo.ocr.models import Issue, extract_full_text, issue_from_dict


@dc.dataclass(frozen=True)
class IssueResult:
    issue_id: str
    cer: float
    wer: float
    predicted_articles: int
    expected_articles: int


@dc.dataclass(frozen=True)
class EvalResult:
    pipeline_name: str
    issues: list[IssueResult]

    @property
    def mean_cer(self) -> float:
        return sum(r.cer for r in self.issues) / len(self.issues) if self.issues else 0.0

    @property
    def mean_wer(self) -> float:
        return sum(r.wer for r in self.issues) / len(self.issues) if self.issues else 0.0


def load_images(issue_dir: pl.Path) -> list[bytes]:
    image_files = sorted(issue_dir.glob("*.jpeg"), key=lambda p: int(p.stem))
    return [f.read_bytes() for f in image_files]


def load_ground_truth(path: pl.Path) -> Issue:
    return issue_from_dict(json.loads(path.read_text()))


def call_ocr_api(api_url: str, images: list[bytes], date: str, source: str = "il_messaggero") -> Issue:
    files = [("files", (f"{i + 1}.jpeg", img, "image/jpeg")) for i, img in enumerate(images)]
    data = {"date": date, "source": source}
    response = httpx.post(f"{api_url}/ocr", files=files, data=data, timeout=300.0)
    response.raise_for_status()
    return issue_from_dict(response.json())


def evaluate_issue(predicted: Issue, expected: Issue) -> IssueResult:
    pred_text = extract_full_text(predicted)
    exp_text = extract_full_text(expected)

    return IssueResult(
        issue_id=predicted.date,
        cer=compute_cer(exp_text, pred_text),
        wer=compute_wer(exp_text, pred_text),
        predicted_articles=len(predicted.articles),
        expected_articles=len(expected.articles),
    )


def evaluate_pipeline(
    api_url: str,
    ground_truth_dir: pl.Path,
    pipeline_name: str = "unknown",
) -> EvalResult:
    issue_dirs = sorted(
        [d for d in ground_truth_dir.iterdir() if d.is_dir() and (d / "ground_truth.json").exists()],
        key=lambda d: d.name,
    )

    results: list[IssueResult] = []
    for issue_dir in issue_dirs:
        images = load_images(issue_dir)
        predicted = call_ocr_api(api_url, images, date=issue_dir.name)
        expected = load_ground_truth(issue_dir / "ground_truth.json")
        results.append(evaluate_issue(predicted, expected))

    return EvalResult(pipeline_name=pipeline_name, issues=results)


def print_eval_results(results: list[EvalResult]) -> None:
    header = f"{'Pipeline':<20} {'CER':>8} {'WER':>8} {'Issues':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.pipeline_name:<20} {r.mean_cer:>8.4f} {r.mean_wer:>8.4f} {len(r.issues):>8}")
