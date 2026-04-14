from __future__ import annotations

import sys

from mausoleo.eval.evaluate import evaluate_all, print_results

if __name__ == "__main__":
    dates = sys.argv[1:] if len(sys.argv) > 1 else None
    results = evaluate_all(dates=dates)
    print_results(results)
