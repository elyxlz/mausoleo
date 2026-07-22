#!/bin/bash
# Caller-side budget measurement: times a full experiment run end-to-end over the
# 6 eval issues (model load + layout + OCR + grouping + IO all included) and reports
# total wall seconds and wall seconds/page. The experiment itself does NOT measure budget.
# Usage: time_experiment.sh experiments/<name>.py
set -e
DATES="1885-06-15 1895-06-15 1910-06-15 1925-06-15 1935-06-15 1952-06-15"
PAGES=42
cd /home/audiogen/mausoleo_di_roma
TORCH_LIBS=$(.venv/bin/python -c "import os,torch;d=os.path.dirname(torch.__file__);print(os.path.join(d,'lib')+':'+os.path.join(d,'..','nvidia','cudnn','lib'))")
start=$(date +%s.%N)
CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH="$TORCH_LIBS:$LD_LIBRARY_PATH" PYTHONPATH=experiments:src .venv/bin/python "$1" $DATES
end=$(date +%s.%N)
awk -v s="$start" -v e="$end" -v p="$PAGES" 'BEGIN{w=e-s; printf "BUDGET wall_seconds=%.1f pages=%d sec_per_page=%.2f (cap 250.0)\n", w, p, w/p}'
