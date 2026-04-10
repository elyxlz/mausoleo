from __future__ import annotations


def compute_cer(reference: str, hypothesis: str) -> float:
    import jiwer

    if not reference:
        return 0.0 if not hypothesis else 1.0
    return jiwer.cer(reference, hypothesis)  # type: ignore[no-any-return]


def compute_wer(reference: str, hypothesis: str) -> float:
    import jiwer

    if not reference:
        return 0.0 if not hypothesis else 1.0
    return jiwer.wer(reference, hypothesis)  # type: ignore[no-any-return]


def compute_kendalls_tau(predicted_order: list[int], expected_order: list[int]) -> float:
    if len(predicted_order) != len(expected_order):
        raise ValueError("order lists must have the same length")
    n = len(predicted_order)
    if n < 2:
        return 1.0

    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            pred_diff = predicted_order[i] - predicted_order[j]
            exp_diff = expected_order[i] - expected_order[j]
            product = pred_diff * exp_diff
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1

    total = concordant + discordant
    if total == 0:
        return 1.0
    return (concordant - discordant) / total
