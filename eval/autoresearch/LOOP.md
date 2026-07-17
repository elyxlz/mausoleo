# Loop state — live queue for the autoresearch loop

Rewritten by the loop every iteration. Rules of the program live in `program.md` (+ `registry.md`); this file is only the CURRENT state and queue.

## Standing context
- Run everything locally on ripperred (`.venv/bin/python`); GPU1 preferred (`CUDA_VISIBLE_DEVICES=1`, eve holds ~350MB on GPU0).
- Fable spend limit hit → spawn subagents with `model: "opus"`.
- Elio's decisions (final): eval GT set = 6 issues; NO issue-level holdout; 1925 Il Meridiano accepted.
- Tentative GTs (1895/129u, 1925/108u, 1935/256u, 1952/197u) delivered in `eval/tentative_gt/` — awaiting Elio's human review; do NOT promote to `eval/ground_truth/`.
- plan/01 ship bar + plan/02 corpus-v0-early stance await Elio's sign-off — no corpus run without it.
- Commit and push as you go; log every result to `log.jsonl` with a mechanism line; update `registry.md` every iteration.

## In flight
- **paddle env install** (task bjihh6qy5): `~/paddle_env` with paddlepaddle-gpu 3.2.0 cu126 + paddleocr. Dead-end HF caches pruned (126G free).
- **exp_158** written: `experiments/exp_158_ppdoclayout_paddle.py` — PP-DocLayoutV3 layout (paddle_env subprocess) + PaddleOCR-VL-1.6 vllm OCR + MergePages(title_class_headlines, squeeze_char_runs). ONE variable vs exp_157: region source (PP-DocLayoutV3 vs DocLayout-YOLO).

## Queue
1. When paddle install completes: verify `~/paddle_env/bin/python -c "import paddle; paddle.utils.run_check()"` and `from paddleocr import LayoutDetection`; smoke-test `LayoutDetection(model_name="PP-DocLayoutV3")` on `eval/ground_truth/1885-06-15/1.jpeg` (downloads ~124M model; check emitted labels against exp_158's TITLE_LABELS/TEXT_LABELS, adjust if they differ).
2. Run `CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/exp_158_ppdoclayout_paddle.py 1885-06-15 1910-06-15` (background).
3. Evaluate: `scripts/research.py eval exp_158_ppdoclayout_paddle` vs exp_157 (0.4284 avg; recall 0.488/0.358). Inspect concrete predictions vs GT; audit giant blobs; holdout; probe on 1943-07-15 (`research.py probe exp_158_ppdoclayout_paddle_1943-07-15` vs exp_157: lexicon 0.6757, high-rep 0.085). Accept per program.md floors (single-source ≥0.005 both dates). Log + registry F3 either way; commit+push.
4. If paddle is a dependency wall: log exact failure + unblock condition in registry F3; move to F1 abandon-class filtering (`experiments/exp_159`: exp_157 with "abandon" removed from YoloCrop text_classes — one variable).
5. Backlog after that: horizontal_overlap lever (F1); prune5 precision filtering (F4, oracle-only); Paddle as prune5 diversity source (F4).
