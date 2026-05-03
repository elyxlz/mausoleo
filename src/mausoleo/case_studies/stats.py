"""Sign test + Cohen's kappa helpers."""
from __future__ import annotations

import math
import typing as tp

try:
    from scipy.stats import binomtest  # type: ignore
except Exception:
    binomtest = None  # type: ignore


def sign_test(pairs: list[tuple[float, float]]) -> dict[str, tp.Any]:
    """Two-sided paired sign test on (mausoleo, baseline) pairs.

    Returns wins / losses / ties and a two-sided binomial p-value over the
    non-tied pairs.
    """
    wins = sum(1 for m, b in pairs if m > b)
    losses = sum(1 for m, b in pairs if m < b)
    ties = sum(1 for m, b in pairs if m == b)
    n = wins + losses
    if n == 0:
        return {"wins": wins, "losses": losses, "ties": ties, "n_decisive": 0, "p_value": None}
    if binomtest is not None:
        try:
            p = float(binomtest(wins, n, p=0.5, alternative="two-sided").pvalue)
        except Exception:
            p = _binom_two_sided(wins, n)
    else:
        p = _binom_two_sided(wins, n)
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "n_decisive": n,
        "p_value": p,
    }


def _binom_two_sided(k: int, n: int) -> float:
    """Manual two-sided binomial p-value at p=0.5."""
    # Two-sided = 2 * min-tail (clipped at 1) under symmetric null.
    def cdf(x: int) -> float:
        return sum(math.comb(n, i) * 0.5 ** n for i in range(x + 1))
    lower = cdf(min(k, n - k))
    return min(1.0, 2.0 * lower)


def cohen_kappa(a: list[int], b: list[int], categories: list[int]) -> float:
    """Cohen's kappa for two raters over a finite category set."""
    if len(a) != len(b) or not a:
        return float("nan")
    n = len(a)
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    # marginal probabilities
    pa = {c: a.count(c) / n for c in categories}
    pb = {c: b.count(c) / n for c in categories}
    exp = sum(pa[c] * pb[c] for c in categories)
    if exp >= 1.0:
        return 1.0
    return (obs - exp) / (1.0 - exp)


def discretise_score(s: float) -> int:
    """Round to nearest integer in 0..5 (kappa needs ordinal categories)."""
    return max(0, min(5, int(round(s))))
