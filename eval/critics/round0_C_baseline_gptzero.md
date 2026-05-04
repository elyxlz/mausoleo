# Stage C Round 0 baseline — GPTZero on R95+ (commit 36b2df5)

## Result

- **completely_generated_prob**: 0.7752
- **confidence_score**: 0.7756
- **class_probabilities**: human 0.224, ai 0.776, mixed 0.0
- **predicted_class**: ai
- **result_message**: "Our detector is moderately confident that the text is written by AI."

## Comparison vs R95 baseline

- R95 (pre-appendix expansion): 0.929
- R95+ (post-appendix expansion + 4 figures): **0.775**
- Δ = **-0.154** natural improvement from appendix expansion + figure embedding

## Ship threshold check

- Target: GPTZero ≤ 0.30 SHIP-ready / ≤ 0.20 SHIP-close-out
- Current: 0.7752 — above SHIP threshold by ~0.48
- VERDICT: NOT SHIP-ready. Continue Strategy 2+3+4 ladder.

## Quota

- 1/7 GPTZero scans burned in 24h.

## Result file

`/tmp/gptzero_round0_C/baseline.json`
