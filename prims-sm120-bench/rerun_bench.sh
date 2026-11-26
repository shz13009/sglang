#!/usr/bin/env bash
# Step 3 regression (switch unset) + step 4 microbenchmark on the current HEAD.
source ~/code/prims-sm120-bench/env.sh
unset SGLANG_ENABLE_FLASHINFER_SM120_FP8_RAGGED_PREFILL
B=~/code/prims-sm120-bench
echo "HEAD $(git rev-parse --short=10 HEAD)"
echo "== regression"
python test/registered/attention/unittests/dense/test_flashinfer.py -v > $B/logs/rerun_step3_regression.log 2>&1
grep -E "^(ok|FAIL|ERROR)|^Ran|^OK|^FAILED" $B/logs/rerun_step3_regression.log
echo "== microbench"
cd $B
python ragged_prefill_bench.py bench_results_c5785136.json > logs/rerun_step4_microbench.log 2>&1
echo "bench exit $?"
grep -E "error|unavailable" logs/rerun_step4_microbench.log | grep -v nvcc | cut -c1-300 | head
python summarize_bench.py bench_results_c5785136.json > bench_tables_c5785136.md
cat bench_tables_c5785136.md
