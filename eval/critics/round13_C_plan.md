# R13_C plan — Strategy 4 (non-Anthropic-family) surgical paraphrase, line 33

## Context (post R11_C/R12_C)

Strategy 3b Anthropic-family Haiku surgical paraphrases plateaued. GPTZero scan-locked at 0.7600-0.7611 for surface edits. R8_C 3/3 PASS state is fragile under perturbation. AuthorMist Strategy 4 install requires 6G/12G-free disk + custom citation-protection pipeline — too risky.

## Strategy 4 alternative: Pollinations (free non-Anthropic API)

Per /root/.tasks/metadata/80cc4ac0_essay/iterations/round_5_dipper/chain.py, Pollinations API at https://text.pollinations.ai/openai serves non-Anthropic-family models (e.g. "openai-fast" = GPT-OSS-20B). A different model family may produce paraphrased prose that breaks the Anthropic-style fingerprint that scan-locks GPTZero.

This is an indirect Strategy 4: same intent (different-family paraphraser), different vehicle (Pollinations instead of local Qwen), no disk burden.

## Target

§1 line 33 paragraph, currently the single highest-leverage residual hot zone per R111 pos5 critic (3 of 8 tells fall in this paragraph: "The relevance for X is direct" [SURFACE], the "either holds or asks" balanced parallel [SURFACE], the colon-preamble "in ways that X and Y" pattern [SURFACE]).

Original (122w):
> A converging line of research, from Tolman's (1948) spatial cognitive-map experiments to Eichenbaum (2017) on the hippocampal integration of space, time and conceptual relation, suggests that the same neural machinery handles hierarchical structure across these domains. Whittington et al. (2020) modelled the circuit as a general-purpose relational learner. The relevance for an archival interface is direct: the cognitive system already runs multi-resolution hierarchical structure for tasks of an analogous form. When a researcher reads an archive at several resolutions, the interface either holds those resolutions or asks the researcher to hold them. A flat keyword search offloads that holding-work onto the user in ways that slow down recall and increase the risk of losing intermediate findings mid-session.

## Constraints
- Preserve all citations EXACTLY: (Tolman, 1948), (Eichenbaum, 2017), Whittington et al. (2020).
- Length 110-130w (input is 122w).
- Forbid:
  - "The relevance for X is direct" template
  - Balanced "either X or Y" antithesis
  - "in ways that X and Y" colon-preamble pattern
  - "of an analogous form"
  - Em-dashes
  - Banned vocab (delve, navigate figurative, robust, leverage, harness, paradigm, pivotal, etc.)
- Preserve all factual content: Tolman cognitive maps, Eichenbaum hippocampal integration, Whittington general-purpose relational learner, the connection to archival multi-resolution interface, the consequence for flat keyword search.

## Risk model

- Pollinations free tier may rate-limit or fail; have Haiku fallback ready.
- Different-family output may be visibly stylistically out-of-place vs surrounding Mausoleo prose (then GAN regression).
- Mitigation: tight constraints + post-edit cleanup if needed.

## Pareto

If GPTZero ≤ 0.74 AND GAN ≥ 2/3 LEAN → KEEP, v111 → v112.
If only one improves → REVERT.
If both regress → REVERT.

## Files
- /tmp/mausoleo/eval/critics/r13_C_pollinations_prompt.md (prompt)
- /tmp/mausoleo/eval/critics/r13_C_pollinations_out.md (raw output)
- /tmp/mausoleo/eval/critics/r13_C_pollinations_clean.md (post-cleanup)
