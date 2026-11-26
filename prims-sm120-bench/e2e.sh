#!/usr/bin/env bash
# Step 4 end-to-end smoke: Qwen3-8B, flashinfer backend, fp8_e4m3 KV cache,
# SGLANG_ENABLE_FLASHINFER_SM120_FP8_RAGGED_PREFILL off vs on.
# Usage: bash e2e.sh off|on
set -u
source ~/code/prims-sm120-bench/env.sh
MODE=$1
OUT=~/code/prims-sm120-bench/e2e/$MODE
mkdir -p "$OUT"
echo "HEAD $(git rev-parse --short=10 HEAD) MODE=$MODE"
PORT=30000
MODEL=Qwen/Qwen3-8B

if [ "$MODE" = on ]; then
  export SGLANG_ENABLE_FLASHINFER_SM120_FP8_RAGGED_PREFILL=1
else
  unset SGLANG_ENABLE_FLASHINFER_SM120_FP8_RAGGED_PREFILL
fi

python -m sglang.launch_server --model-path $MODEL --port $PORT \
  --attention-backend flashinfer --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.85 --random-seed 0 \
  > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT

for i in $(seq 1 600); do
  if curl -sf "http://127.0.0.1:$PORT/health_generate" > /dev/null; then break; fi
  if ! kill -0 $SERVER 2>/dev/null; then echo "SERVER_DIED"; tail -40 "$OUT/server.log"; exit 1; fi
  sleep 2
done
echo "SERVER_READY after ~$((i * 2))s"
grep -E "SM120 FP8 ragged attention enabled|chunked_prefill_size|max_total_num_tokens|KV Cache is allocated" "$OUT/server.log" | cut -c1-300 | head -8

echo "== GSM8K"
python -m sglang.test.few_shot_gsm8k --num-questions 500 --num-shots 5 --parallel 64 \
  --temperature 0 --port $PORT 2>&1 | tee "$OUT/gsm8k.log" | grep -E "Accuracy|Invalid|Latency|throughput"

echo "== prefill-heavy"
# Every measured run flushes the radix cache first and uses its own seed, so no
# prompt is served from cache (run 1 of this script reused seed 42 across C=1/C=4).
flush() { curl -sf -X POST "http://127.0.0.1:$PORT/flush_cache" > /dev/null || echo "flush_cache failed"; }
SEED=100
for L in 8192 16384 32768; do
  # warmup (JIT / cute-dsl compile, CUDA graph capture), discarded
  flush
  python -m sglang.benchmark.serving --backend sglang --port $PORT --dataset-name random-ids \
    --random-input-len $L --random-output-len 1 --random-range-ratio 1.0 \
    --num-prompts 2 --max-concurrency 1 --seed 1 > /dev/null 2>&1
  for C in 1 4; do
    N=$([ $C = 1 ] && echo 8 || echo 16)
    SEED=$((SEED + 1))
    flush
    MARK=$(wc -l < "$OUT/server.log")
    python -m sglang.benchmark.serving --backend sglang --port $PORT --dataset-name random-ids \
      --random-input-len $L --random-output-len 1 --random-range-ratio 1.0 \
      --num-prompts $N --max-concurrency $C --seed $SEED \
      --output-file "$OUT/prefill.jsonl" > "$OUT/prefill_${L}_c${C}.log" 2>&1
    CACHED=$(tail -n +$((MARK + 1)) "$OUT/server.log" | grep -oE "#cached-token: [0-9]+" | awk '{s += $2} END {print s + 0}')
    echo "L=$L C=$C seed=$SEED cached_tokens=$CACHED: $(grep -E "Median TTFT|Mean TTFT|Input token throughput" "$OUT/prefill_${L}_c${C}.log" | tr -s ' ' | tr '\n' ';')"
  done
done
grep -ciE "error|traceback" "$OUT/server.log" | sed 's/^/server.log error-ish lines: /'
echo DONE
