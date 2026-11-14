#!/usr/bin/env bash
# Evaluate base model and a LoRA checkpoint with the same protocol (paired, multi-sample).
#   chapter3/eval_ckpt.sh chapter3/runs/g1-4b-d234-8k/checkpoint-100 [sets] [samples] [limit] [max_tokens]
# Writes chapter3/results/<tag>/ for "base" (only if missing) and the checkpoint; compare with
#   .venv/bin/python chapter3/compare.py chapter3/results/base chapter3/results/<tag>
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CKPT=$(cd "$1" && pwd)
SETS=${2:-d3,d5}
SAMPLES=${3:-4}
LIMIT=${4:-50}
MAXTOK=${5:-32768}
BASE=Qwen/Qwen3-4B
TAG=$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")
cd "$ROOT"

pkill -f "bin/vllm serv[e]" || true  # [e]: never matches a shell whose command line contains this pattern
for i in $(seq 120); do curl -sf localhost:8000/v1/models >/dev/null || break; sleep 1; done
LOG=$ROOT/chapter3/vllm_eval.log
nohup .vllm-venv/bin/vllm serve $BASE --max-model-len 40960 --gpu-memory-utilization 0.85 \
  --enable-lora --max-lora-rank 64 --lora-modules "$TAG=$CKPT" > "$LOG" 2>&1 &
for i in $(seq 1800); do
  curl -sf localhost:8000/v1/models 2>/dev/null | grep -q "\"id\":\"$TAG\"" && break
  sleep 1
done
curl -sf localhost:8000/v1/models | grep -q "\"id\":\"$TAG\"" || { echo "server failed"; tail -30 "$LOG"; exit 1; }

EV=".venv/bin/python -u scripts/eval_openai.py --sets $SETS --limit $LIMIT --samples $SAMPLES --max-tokens $MAXTOK --timeout 3600"
if [ ! -d chapter3/results/base ]; then
  echo "== base =="; $EV --model $BASE --out chapter3/results/base
fi
echo "== $TAG =="; $EV --model "$TAG" --out "chapter3/results/$TAG"
