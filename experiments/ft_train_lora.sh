#!/usr/bin/env bash
set -euo pipefail

FOLD="${1:-1935-06-15}"
GPU="${GPU:-0}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TAG="no${FOLD}"
DATA="${REPO}/eval/autoresearch/paddleft"
OUT="${DATA}/lora_${TAG}"
MERGED="${DATA}/merged_${TAG}"

TRAIN_JSONL="${TRAIN_JSONL:-${DATA}/train_${TAG}.jsonl}"
test -s "${TRAIN_JSONL}" || { echo "missing ${TRAIN_JSONL} — run ft_prepare_data.py ${FOLD} first"; exit 1; }

CUDA_VISIBLE_DEVICES="${GPU}" USE_HF=1 "${REPO}/.venv/bin/swift" sft \
  --model PaddlePaddle/PaddleOCR-VL-1.6 \
  --model_type paddleocr_vl \
  --template paddle_ocr_1_5 \
  --tuner_type lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner false \
  --dataset "${TRAIN_JSONL}" \
  --val_dataset "${DATA}/val_${TAG}.jsonl" \
  --split_dataset_ratio 0 \
  --num_train_epochs ${EPOCHS:-3} \
  --learning_rate 1e-4 \
  --warmup_ratio 0.05 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_length 8192 \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --gradient_checkpointing true \
  --eval_strategy steps \
  --eval_steps 8 \
  --save_steps 8 \
  --save_total_limit 2 \
  --load_best_model_at_end true \
  --logging_steps 1 \
  --dataloader_num_workers 4 \
  --seed 0 \
  --output_dir "${OUT}"

BEST="$("${REPO}/.venv/bin/python" - "${OUT}" <<'PY'
import json, pathlib as pl, sys
run_dirs = sorted(pl.Path(sys.argv[1]).glob("v*"), key=lambda p: p.stat().st_mtime)
checkpoints = sorted(run_dirs[-1].glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
state = json.loads((checkpoints[-1] / "trainer_state.json").read_text())
print(state.get("best_model_checkpoint") or str(checkpoints[-1]))
PY
)"
echo "best adapter checkpoint: ${BEST}"

CUDA_VISIBLE_DEVICES="${GPU}" USE_HF=1 "${REPO}/.venv/bin/swift" export \
  --adapters "${BEST}" \
  --merge_lora true \
  --output_dir "${MERGED}"

echo "merged checkpoint: ${MERGED}"
