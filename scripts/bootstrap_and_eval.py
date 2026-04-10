from __future__ import annotations

import dataclasses as dc
import json
import pathlib as pl

from mausoleo.eval.metrics import compute_cer, compute_wer
from mausoleo.ocr.models import Issue, extract_full_text, issue_from_dict

PREDICTIONS_DIR = pl.Path("eval/predictions")
BOOTSTRAP_DIR = pl.Path("eval/bootstrap_gt")
ISSUE_DATES = ["1885-06-15", "1910-06-15", "1940-04-01"]


def is_clean_prediction(issue: Issue) -> bool:
    for article in issue.articles:
        for para in article.paragraphs:
            t = (para.text or "").strip()
            if t.startswith("{") or t.startswith("```") or t.startswith("---"):
                return False
    return True


def dedup_text(text: str, chunk_size: int = 150) -> str:
    lines = text.split("\n")
    seen: set[str] = set()
    deduped: list[str] = []
    for line in lines:
        normalized = line.strip()
        if not normalized:
            deduped.append(line)
            continue
        if len(normalized) < 20:
            deduped.append(line)
            continue
        key = normalized[:chunk_size]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(line)
    return "\n".join(deduped)


def dedup_issue(issue_data: dict) -> dict:
    for article in issue_data.get("articles", []):
        for para in article.get("paragraphs", []):
            t = para.get("text") or ""
            if len(t) > 2000:
                para["text"] = dedup_text(t)
    return issue_data


def repetition_ratio(text: str, chunk_size: int = 200) -> float:
    if len(text) < chunk_size * 3:
        return 0.0
    chunks = [text[i : i + chunk_size] for i in range(0, len(text) - chunk_size, chunk_size)]
    if not chunks:
        return 0.0
    seen: set[str] = set()
    repeated = 0
    for c in chunks:
        if c in seen:
            repeated += 1
        seen.add(c)
    return repeated / len(chunks)


def load_predictions(issue_date: str) -> list[tuple[str, Issue, str]]:
    results: list[tuple[str, Issue, str]] = []
    for f in sorted(PREDICTIONS_DIR.glob(f"*_{issue_date}.json")):
        try:
            data = json.loads(f.read_text())
            issue = issue_from_dict(data)
            text = extract_full_text(issue)
            if len(text) < 100:
                continue
            config_name = f.stem.replace(f"_{issue_date}", "")
            results.append((config_name, issue, text))
        except Exception:
            continue
    return results


def bootstrap_ground_truth() -> None:
    print("BOOTSTRAPPING GROUND TRUTH")
    print("=" * 60)

    BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)

    for issue_date in ISSUE_DATES:
        all_preds = load_predictions(issue_date)
        clean_preds = [(n, i, t) for n, i, t in all_preds if is_clean_prediction(i)]

        print(f"\n  {issue_date}: {len(all_preds)} total predictions, {len(clean_preds)} clean")
        for name, issue, text in all_preds:
            clean = is_clean_prediction(issue)
            n_arts = len(issue.articles)
            flag = "OK" if clean else "BAD"
            print(f"    [{flag:3s}] {name:<40s} {n_arts:>3d} arts  {len(text):>7d} chars")

        candidates = clean_preds if clean_preds else all_preds
        if not candidates:
            print(f"    NO valid predictions!")
            continue

        deduped_candidates: list[tuple[str, Issue, str]] = []
        for name, issue, text in candidates:
            rep = repetition_ratio(text)
            if rep > 0.3:
                fixed_data = dedup_issue(dc.asdict(issue))
                fixed_issue = issue_from_dict(fixed_data)
                fixed_text = extract_full_text(fixed_issue)
                print(f"    DEDUP {name}: {len(text)}c -> {len(fixed_text)}c (rep={rep:.0%})")
                deduped_candidates.append((name, fixed_issue, fixed_text))
            else:
                deduped_candidates.append((name, issue, text))

        best_idx = 0
        best_score = -1.0
        for i in range(len(deduped_candidates)):
            text_i = deduped_candidates[i][2]
            rep_penalty = repetition_ratio(text_i) * 0.5

            sims = []
            for j in range(len(deduped_candidates)):
                if i == j:
                    continue
                cer = compute_cer(text_i, deduped_candidates[j][2])
                sims.append(1.0 - min(cer, 1.0))
            avg_sim = sum(sims) / len(sims) if sims else 0.0
            length_bonus = min(len(text_i) / 50000, 1.0) * 0.1
            score = avg_sim + length_bonus - rep_penalty
            if score > best_score:
                best_score = score
                best_idx = i

        config_name, best_issue, best_text = deduped_candidates[best_idx]
        rep = repetition_ratio(best_text)

        gt_dir = BOOTSTRAP_DIR / issue_date
        gt_dir.mkdir(parents=True, exist_ok=True)
        gt_path = gt_dir / "ground_truth.json"
        gt_path.write_text(json.dumps(dc.asdict(best_issue), indent=2, ensure_ascii=False))
        print(f"    -> bootstrapped from '{config_name}' (score={best_score:.3f}, {len(best_text)} chars, {len(best_issue.articles)} arts, rep={rep:.0%})")


def evaluate_all() -> None:
    print("\n\nEVALUATION RESULTS")
    print("=" * 60)

    results: list[tuple[str, str, float, float, int, int, bool]] = []

    for issue_date in ISSUE_DATES:
        gt_path = BOOTSTRAP_DIR / issue_date / "ground_truth.json"
        if not gt_path.exists():
            continue
        gt_issue = issue_from_dict(json.loads(gt_path.read_text()))
        gt_text = extract_full_text(gt_issue)

        preds = load_predictions(issue_date)
        for config_name, pred_issue, pred_text in preds:
            cer = compute_cer(gt_text, pred_text)
            wer = compute_wer(gt_text, pred_text)
            clean = is_clean_prediction(pred_issue)
            results.append((config_name, issue_date, cer, wer, len(pred_issue.articles), len(pred_text), clean))

    print(f"\n{'Config':<40s} {'Issue':<12} {'CER':>8} {'WER':>8} {'Arts':>5} {'Chars':>7} {'Clean':>5}")
    print("-" * 90)
    for config_name, issue_date, cer, wer, n_articles, n_chars, clean in sorted(results, key=lambda r: (r[1], r[2])):
        flag = "Y" if clean else "N"
        print(f"{config_name:<40s} {issue_date:<12} {cer:>8.4f} {wer:>8.4f} {n_articles:>5} {n_chars:>7} {flag:>5}")

    config_scores: dict[str, list[tuple[float, float, bool]]] = {}
    for config_name, _, cer, wer, _, _, clean in results:
        config_scores.setdefault(config_name, []).append((cer, wer, clean))

    print(f"\n\nLEADERBOARD (avg CER across issues)")
    print("=" * 65)
    ranked = sorted(config_scores.items(), key=lambda x: sum(c for c, _, _ in x[1]) / len(x[1]))
    for rank, (config_name, scores) in enumerate(ranked, 1):
        avg_cer = sum(c for c, _, _ in scores) / len(scores)
        avg_wer = sum(w for _, w, _ in scores) / len(scores)
        n_issues = len(scores)
        all_clean = all(cl for _, _, cl in scores)
        flag = "" if all_clean else " *has bad formatting*"
        print(f"  {rank:>2}. {config_name:<40s} CER={avg_cer:.4f}  WER={avg_wer:.4f}  ({n_issues} issues){flag}")

    leaderboard = pl.Path("eval/leaderboard.txt")
    with open(leaderboard, "w") as f:
        f.write("OCR Pipeline Leaderboard (bootstrapped eval)\n")
        f.write("=" * 65 + "\n")
        f.write(f"{'Rank':>4}  {'Config':<40s} {'CER':>8} {'WER':>8} {'Issues':>6}\n")
        f.write("-" * 70 + "\n")
        for rank, (config_name, scores) in enumerate(ranked, 1):
            avg_cer = sum(c for c, _, _ in scores) / len(scores)
            avg_wer = sum(w for _, w, _ in scores) / len(scores)
            f.write(f"{rank:>4}  {config_name:<40s} {avg_cer:>8.4f} {avg_wer:>8.4f} {len(scores):>6}\n")
    print(f"\nSaved to {leaderboard}")


if __name__ == "__main__":
    bootstrap_ground_truth()
    evaluate_all()
