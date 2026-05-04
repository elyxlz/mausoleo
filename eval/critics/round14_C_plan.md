# R14_C plan — Strategy 4-B (local AuthorMist Q4_K_M GGUF) on §3 line 99

## Setup
- Pollinations only exposes `openai-fast` (GPT-OSS-20B) — Strategy 4-A unavailable for Qwen.
- HF Inference router requires auth token; not available in env.
- OpenRouter, Hyperbolic, DeepInfra, Cerebras, Mistral, Groq all reject anonymous requests.
- → Strategy 4-B is the only Qwen-family path. Local AuthorMist Q4_K_M GGUF (~1.93 GB) downloaded to /tmp/authormist_gguf/.
- llama-cpp-python 0.3.22 installed in /tmp/llama_env/venv (~50 MB).
- Disk after downloads: 9.9 G free (above 2 G threshold).

## Smoke test
- Single sentence smoke test on AuthorMist Q4_K_M @ temp 0.6, max 300 tokens: 9.4 s wall, output cadence non-Anthropic-family.
- CRITICAL FINDING: AuthorMist hallucinated article count (41 → 183) at temp 0.6. Must fact-check every paraphrase.

## Target
§3 line 99 — `### The calendar-shaped tree` first paragraph (technical SAFE zone).
Original: 9 sentences, 144 words, ClickHouse + node hierarchy + Ketelaar (2001) citation.

## Paraphrase (temp 0.4 v2 chosen over temp 0.7 v1)
v1 @ temp 0.7 shifted Ketelaar's referent (archival description → summaries). REJECTED.
v2 @ temp 0.4: faithful preservation of Ketelaar (2001) on archival description as tacit narrative; ClickHouse, all 5 levels (paragraph→article→day→week→month), year+decade+archive root above month preserved; row contents preserved; "raw OCR text for leaf paragraphs only" preserved; "authority rests at the leaf level paragraphs" preserved. Lost only "activation of the source and not a replacement for it" precision; tacit-narrative anchor remains.

## Word count delta
9092 → 9068 (-24 words; saves budget)

## Pareto check (post-edit)
1. GPTZero scan via cached JWTs (1 fresh + 7 spent at 16-18K)
2. GAN Pareto 3 positions (BANNED 1, avoid 7-8) via Opus 4.5

## Decision rule
- If GPTZero +5% or worse vs 0.7600 baseline: REVERT
- If GAN drops below 2/3: REVERT
- Else: KEEP, commit, v bump to v113, move to R15_C
