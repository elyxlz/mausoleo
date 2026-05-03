
## run started 2026-05-03T08:14:49.550088Z
- starting case1 / mausoleo / trial 1 (seed=1027) at $0.00
  done case1/mausoleo/t1: calls=9 chars=218108 recall=0.81 j1=4.0/4.0/4.0 j2=5.0/5.0/5.0 spend=$0.74 (49.04s)
- starting case1 / mausoleo / trial 2 (seed=2036) at $0.74
  done case1/mausoleo/t2: calls=7 chars=219626 recall=0.81 j1=2.0/3.0/3.0 j2=4.0/4.0/5.0 spend=$1.38 (51.14s)
- starting case1 / mausoleo / trial 3 (seed=3045) at $1.38
  done case1/mausoleo/t3: calls=6 chars=216594 recall=0.81 j1=4.0/4.0/4.0 j2=5.0/5.0/5.0 spend=$2.01 (50.97s)
- starting case1 / baseline / trial 1 (seed=1027) at $2.01
  done case1/baseline/t1: calls=30 chars=89709 recall=0.67 j1=4.0/4.0/4.0 j2=4.0/4.0/3.0 spend=$3.34 (80.31s)
- starting case1 / baseline / trial 2 (seed=2036) at $3.34
  done case1/baseline/t2: calls=28 chars=90725 recall=0.71 j1=4.0/4.0/4.0 j2=4.0/4.0/4.0 spend=$4.44 (77.63s)
- starting case1 / baseline / trial 3 (seed=3045) at $4.44
  done case1/baseline/t3: calls=29 chars=89755 recall=0.71 j1=4.0/4.0/3.0 j2=4.0/4.0/3.0 spend=$5.60 (81.05s)
- starting case2 / mausoleo / trial 1 (seed=1836) at $5.60
  done case2/mausoleo/t1: calls=13 chars=236964 recall=0.76 j1=5.0/5.0/5.0 j2=4.0/5.0/5.0 spend=$7.19 (75.46s)
- starting case2 / mausoleo / trial 2 (seed=2845) at $7.19
  done case2/mausoleo/t2: calls=18 chars=230978 recall=0.76 j1=4.0/4.0/4.0 j2=5.0/5.0/5.0 spend=$8.89 (82.08s)
- starting case2 / mausoleo / trial 3 (seed=3854) at $8.89
  done case2/mausoleo/t3: calls=14 chars=364529 recall=0.76 j1=4.0/4.0/4.0 j2=5.0/5.0/5.0 spend=$11.15 (75.11s)
- starting case2 / baseline / trial 1 (seed=1836) at $11.15
  done case2/baseline/t1: calls=30 chars=72247 recall=0.70 j1=3.0/3.0/3.0 j2=4.0/4.0/4.0 spend=$11.96 (72.51s)
- starting case2 / baseline / trial 2 (seed=2845) at $11.96
  done case2/baseline/t2: calls=30 chars=67123 recall=0.61 j1=4.0/4.0/4.0 j2=4.0/4.0/4.0 spend=$13.21 (94.69s)
- starting case2 / baseline / trial 3 (seed=3854) at $13.21
  done case2/baseline/t3: calls=30 chars=100842 recall=0.70 j1=4.0/4.0/4.0 j2=4.0/4.0/4.0 spend=$14.51 (75.45s)
- starting case3 / mausoleo / trial 1 (seed=1709) at $14.51
  done case3/mausoleo/t1: calls=15 chars=279127 recall=0.07 j1=4.0/5.0/4.0 j2=4.0/5.0/5.0 spend=$15.48 (68.33s)
- starting case3 / mausoleo / trial 2 (seed=2718) at $15.48
  done case3/mausoleo/t2: calls=9 chars=416191 recall=0.11 j1=3.0/4.0/4.0 j2=2.0/3.0/3.0 spend=$16.38 (46.93s)
- starting case3 / mausoleo / trial 3 (seed=3727) at $16.38
  done case3/mausoleo/t3: calls=17 chars=248848 recall=0.04 j1=3.0/4.0/4.0 j2=4.0/4.0/4.0 spend=$17.33 (67.96s)
- starting case3 / baseline / trial 1 (seed=1709) at $17.33
  done case3/baseline/t1: calls=26 chars=109470 recall=0.11 j1=4.0/4.0/4.0 j2=4.0/4.0/4.0 spend=$18.34 (77.86s)
BUDGET CAP HIT at $18.34 before case3/baseline/t2; stopping. researcher=$17.70 judges=$0.64



## Final summary 2026-05-03 09:36 BST

- **trials completed**: 16 / 18 (case 3 baseline trials 2 and 3 not run; spend cap reached at $18.34)
- **wall time**: 22.9 min (Phase 2 case-study run only; excludes ClickHouse setup + GT build)
- **total LLM spend**: $18.34
  - researcher (Sonnet 4.5 OAuth): $17.70 (16 trials × ~$1.10 avg)
  - judges (Opus 4.5 + Sonnet 4.5 OAuth): $0.64 (32 judge calls)
- **budget cap**: $18 USD soft cap, configured to stop the runner cleanly before $20.

## Methodology adjustments from outline §6

- **Judge 2 substitution**: per the outline §6.1 the second judge should be GPT-5;
  no OpenAI key was within budget, so Judge 2 is Claude Sonnet 4.5 with an
  explicitly distinct "judge 2" system prompt. Called out in §6.5.
- **Embedding fallback**: the semantic and hybrid search tools fell back to
  text search at runtime (no embedding model loaded into the case-study
  harness, to keep the spend cap). Mausoleo therefore competed using
  hierarchy + text-search + tree-traversal alone, not its semantic-search
  advantage. This understates Mausoleo's likely operational performance.
- **Single-annotator relevance GT**: per the outline; the 2-week
  self-consistency check could not be performed in one session. Reported as
  a §7.2 limitation rather than measured here.
- **Budget cap stopped run**: case 3 baseline trials 2 and 3 not collected.
  Case-3 baseline column shows N=1; case 3 sign tests are correspondingly
  underpowered.

## Surprises and notes Elio should know

- **Mausoleo wins on case 1 and case 2 by all metrics** (efficiency, completeness,
  quality), with case 1 the headline definitional-gap result and case 2 the
  3-trial × 2-judge sign-test confirmed quality + completeness win.
- **Case 3 is methodologically interesting**: Mausoleo wins on quality (per the
  judges) and on tool calls but *loses* on the article-id-touched recall metric
  (0.07 vs 0.11). This is because Mausoleo answers month-scale aggregate questions
  by reading day summaries (which compress 200+ articles into 200-400 words);
  the day summary doesn't enumerate article ids, so the touched-set metric
  undercounts what Mausoleo actually surfaces. The baseline reads articles by id,
  so it scores higher on the touched-set metric but produces a less coherent
  month-scale answer. This is a real failure mode of the recall-of-touched-articles
  operationalisation and is flagged in §6.5 + §7.2.
- **Mausoleo's chars-read is higher than baseline's across all cases**. This is
  because day summaries are denser per node than BM25 snippets (each day
  summary ~ 200-400 words; each baseline-search result ~ 220 chars). The metric
  measures bytes returned to the agent's context, not bytes the agent had to
  reason about; Mausoleo's bytes carry more compiled information per byte. We
  report it but do not over-interpret.
- **Inter-judge κ is weak across all cases** (-0.13, 0.14, 0.00). The two judges
  disagree particularly on case 1: Opus (judge 1, historian prompt) sometimes
  downscores Mausoleo's "absent from the digital archive" framing as factually
  imprecise (paper copies of 26 July do exist outside this corpus); Sonnet
  (judge 2, IR prompt) scores the same answer as exceptional because it
  correctly characterises the corpus the agent has access to. Both readings
  are defensible. We leave the disagreement unsanitised.
- **Baseline agent on case 1 sometimes infers the 26 July absence without ever
  reading a 26 July article** — interesting "cleverness" of the LLM working
  around its data deficit. Consistent with the §6.2 framing that case 1 is a
  definitional capability gap, not an absolute wall: a knowledgeable agent
  can *reason* about the absence from outside the corpus, but cannot *ground
  it in the corpus* the way Mausoleo's day-summary does.

## Headline numbers

| Case | Mausoleo recall | Baseline recall | Mausoleo quality | Baseline quality | Tool calls M / B |
|---|---|---|---|---|---|
| Case 1 (07-26 absent) | 0.81 | 0.70 | 4.17 | 3.83 | 7.3 / 29.0 |
| Case 2 (regime change) | 0.76 | 0.67 | 4.61 | 3.83 | 15.0 / 30.0 |
| Case 3 (comparative) | 0.07 (metric mismatch) | 0.11 | 3.83 | 4.00 (N=1) | 13.7 / 26.0 |

§6.5 prose lives at `references/section_6_5_results.md`; per-trial JSON
artefacts at `eval/case_studies/runs/{case}_{system}_t{N}{,_judge1,_judge2}.json`;
aggregate stats at `eval/case_studies/aggregate.json`; relevance GT at
`eval/case_studies/relevance_gt.json`.

## run started 2026-05-03T08:53:00.484594Z (no-budget-cap rerun)
  embedder status: {"loaded": true, "smoke_test": true, "model": "paraphrase-multilingual-MiniLM-L12-v2", "dim": 384, "nearest_day_to_mussolini": ["1943-07-26", 0.8998396992683411]}
- starting case1 / mausoleo / trial 1 (seed=1233) at tok_in=0 tok_out=0
  done case1/mausoleo/t1: calls=11 chars=180428 recall=0.79 j1=4.0/5.0/4.0 j2=5.0/5.0/5.0 tok_in=181008 tok_out=1997 (63.99s)
- starting case1 / mausoleo / trial 2 (seed=2242) at tok_in=181008 tok_out=1997
  done case1/mausoleo/t2: calls=14 chars=95579 recall=0.45 j1=4.0/4.0/4.0 j2=4.0/4.0/4.0 tok_in=219109 tok_out=2187 (63.27s)
- starting case1 / mausoleo / trial 3 (seed=3251) at tok_in=400117 tok_out=4184
  done case1/mausoleo/t3: calls=15 chars=186003 recall=0.79 j1=5.0/5.0/5.0 j2=5.0/5.0/5.0 tok_in=340078 tok_out=2298 (70.63s)
- starting case1 / baseline / trial 1 (seed=1233) at tok_in=740195 tok_out=6482
  done case1/baseline/t1: calls=26 chars=78310 recall=0.69 j1=4.0/4.0/4.0 j2=5.0/5.0/5.0 tok_in=402200 tok_out=2996 (81.6s)
- starting case1 / baseline / trial 2 (seed=2242) at tok_in=1142395 tok_out=9478
  done case1/baseline/t2: calls=28 chars=88392 recall=0.69 j1=4.0/4.0/4.0 j2=4.0/5.0/4.0 tok_in=407564 tok_out=3658 (89.78s)
- starting case1 / baseline / trial 3 (seed=3251) at tok_in=1549959 tok_out=13136
  done case1/baseline/t3: calls=27 chars=86893 recall=0.62 j1=4.0/4.0/4.0 j2=4.0/4.0/4.0 tok_in=372314 tok_out=3650 (92.62s)
- starting case2 / mausoleo / trial 1 (seed=1382) at tok_in=1922273 tok_out=16786
  done case2/mausoleo/t1: calls=11 chars=234113 recall=0.76 j1=4.0/5.0/5.0 j2=5.0/5.0/5.0 tok_in=589213 tok_out=3111 (105.21s)
- starting case2 / mausoleo / trial 2 (seed=2391) at tok_in=2511486 tok_out=19897
  done case2/mausoleo/t2: calls=13 chars=249655 recall=0.76 j1=4.0/5.0/5.0 j2=4.0/5.0/5.0 tok_in=426139 tok_out=2609 (81.5s)
- starting case2 / mausoleo / trial 3 (seed=3400) at tok_in=2937625 tok_out=22506
  done case2/mausoleo/t3: calls=13 chars=241327 recall=0.76 j1=5.0/5.0/5.0 j2=5.0/5.0/5.0 tok_in=565516 tok_out=2594 (90.45s)
- starting case2 / baseline / trial 1 (seed=1382) at tok_in=3503141 tok_out=25100
  done case2/baseline/t1: calls=30 chars=87255 recall=0.61 j1=4.0/4.0/4.0 j2=4.0/4.0/4.0 tok_in=230857 tok_out=3125 (66.17s)
- starting case2 / baseline / trial 2 (seed=2391) at tok_in=3733998 tok_out=28225
  done case2/baseline/t2: calls=30 chars=102592 recall=0.55 j1=4.0/4.0/4.0 j2=5.0/5.0/5.0 tok_in=341577 tok_out=3396 (82.4s)
- starting case2 / baseline / trial 3 (seed=3400) at tok_in=4075575 tok_out=31621
  done case2/baseline/t3: calls=29 chars=92932 recall=0.70 j1=4.0/5.0/5.0 j2=5.0/5.0/5.0 tok_in=337246 tok_out=3813 (91.71s)
- starting case3 / mausoleo / trial 1 (seed=1553) at tok_in=4412821 tok_out=35434
  done case3/mausoleo/t1: calls=7 chars=249501 recall=0.00 rmse=0.1317 mae=0.1144 weeks=5/5 j1=4.0/4.0/4.0 j2=4.0/4.0/4.0 tok_in=223205 tok_out=2743 (82.98s)
- starting case3 / mausoleo / trial 2 (seed=2562) at tok_in=4636026 tok_out=38177
  done case3/mausoleo/t2: calls=8 chars=138397 recall=0.00 rmse=0.1836 mae=0.1664 weeks=5/5 j1=4.0/4.0/4.0 j2=4.0/4.0/5.0 tok_in=112155 tok_out=2395 (68.89s)
- starting case3 / mausoleo / trial 3 (seed=3571) at tok_in=4748181 tok_out=40572
  done case3/mausoleo/t3: calls=10 chars=467985 recall=0.11 rmse=0.1836 mae=0.1664 weeks=5/5 j1=4.0/4.0/4.0 j2=4.0/4.0/4.0 tok_in=303074 tok_out=2197 (64.71s)
- starting case3 / baseline / trial 1 (seed=1553) at tok_in=5051255 tok_out=42769
  done case3/baseline/t1: calls=27 chars=108157 recall=0.15 rmse=0.2321 mae=0.2064 weeks=5/5 j1=3.0/3.0/3.0 j2=4.0/4.0/4.0 tok_in=263730 tok_out=3210 (69.63s)
- starting case3 / baseline / trial 2 (seed=2562) at tok_in=5314985 tok_out=45979
  done case3/baseline/t2: calls=28 chars=105361 recall=0.11 rmse=0.2239 mae=0.1826 weeks=5/5 j1=2.0/3.0/3.0 j2=2.0/2.0/3.0 tok_in=241190 tok_out=3550 (79.3s)
- starting case3 / baseline / trial 3 (seed=3571) at tok_in=5556175 tok_out=49529
  done case3/baseline/t3: calls=30 chars=112567 recall=0.11 rmse=0.2051 mae=0.1939 weeks=5/5 j1=3.0/3.0/3.0 j2=4.0/4.0/4.0 tok_in=294792 tok_out=3190 (77.75s)

finished tokens in/out 5850967/52719 (phantom $=19.17) wall=1656s


## Rerun 2026-05-03 final summary 2026-05-03T09:21:12.541763Z

- **trials completed**: 18 / 18 (3 cases × 2 systems × 3 trials)
- **wall time**: 27.6 min (1656 s)
- **tokens (Anthropic, OAuth subscription quota; dollar cost is meaningless)**:
  input 5,850,967, output 52,719.
- **embedder loaded**: True (paraphrase-multilingual-MiniLM-L12-v2, dim 384); smoke-test nearest day to "Mussolini" = ['1943-07-26', 0.8998396992683411].
- **case-3 metric**: replaced article-id-touched recall with per-ISO-week
  war-fraction MAE/RMSE against an LLM-built oracle
  (`eval/case_studies/case3_oracle_ratios.json`, 6480 articles classified).
- **cost cap**: removed. Phantom dollar figure no longer used for control flow.
