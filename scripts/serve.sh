#!/usr/bin/env bash
# Swap the local vLLM server to MODEL and block until it reports that model id.
#   scripts/serve.sh Qwen/Qwen3-4B [max_model_len] [log]
set -u
MODEL=$1; LEN=${2:-40960}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOG=${3:-$ROOT/vllm_$(basename "$MODEL").log}

pkill -f "bin/vllm serv[e]"
for i in $(seq 120); do
  curl -sf localhost:8000/v1/models >/dev/null || { pgrep -f "bin/vllm serv[e]" >/dev/null || break; }
  sleep 1
done
curl -sf localhost:8000/v1/models >/dev/null && { echo "COULD NOT STOP OLD SERVER"; exit 1; }

nohup "$ROOT/.vllm-venv/bin/vllm" serve "$MODEL" --max-model-len "$LEN" --gpu-memory-utilization 0.85 > "$LOG" 2>&1 &
PID=$!
for i in $(seq 1800); do
  curl -sf localhost:8000/v1/models 2>/dev/null | grep -q "\"id\":\"$MODEL\"" && { echo "ready: $MODEL (log $LOG)"; exit 0; }
  kill -0 $PID 2>/dev/null || { echo "SERVER DIED: $MODEL"; tail -30 "$LOG"; exit 1; }
  sleep 1
done
echo "SERVER TIMEOUT: $MODEL"; exit 1
