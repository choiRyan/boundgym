#!/usr/bin/env bash
# Replication eval: base vs a LoRA checkpoint on FRESH held-out levels (levels_rep/, seeds >= 2,000,000, never
# used for training, checkpoint selection or any earlier eval). Same protocol as eval_ckpt.sh.
#   chapter3/eval_rep.sh chapter3/runs/g4-4b-d234-16k-nomask/checkpoint-100
# Writes chapter3/results_rep/{base,<tag>}/ (kept apart from chapter3/results: different levels, never pair them).
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CKPT=$(cd "$1" && pwd)
BASE=Qwen/Qwen3-4B
TAG=$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")
cd "$ROOT"
for i in $(seq 120); do curl -sf localhost:8000/v1/models >/dev/null || break; sleep 1; done
LOG=$ROOT/chapter3/vllm_eval.log
nohup .vllm-venv/bin/vllm serve $BASE --max-model-len 40960 --gpu-memory-utilization 0.85 \
  --enable-lora --max-lora-rank 64 --lora-modules "$TAG=$CKPT" > "$LOG" 2>&1 &
echo $! > chapter3/vllm_eval.pid
for i in $(seq 1800); do
  curl -sf localhost:8000/v1/models 2>/dev/null | grep -q "\"id\":\"$TAG\"" && break
  sleep 1
done
curl -sf localhost:8000/v1/models | grep -q "\"id\":\"$TAG\"" || { echo "server failed"; tail -30 "$LOG"; exit 1; }
EV=".venv/bin/python -u scripts/eval_openai.py --levels levels_rep --sets d3,d5 --limit 50 --samples 4 --max-tokens 32768 --timeout 3600"
[ -d chapter3/results_rep/base ] || { echo "== base =="; $EV --model $BASE --out chapter3/results_rep/base; }
echo "== $TAG =="; $EV --model "$TAG" --out "chapter3/results_rep/$TAG"
kill "$(cat chapter3/vllm_eval.pid)" && rm -f chapter3/vllm_eval.pid
