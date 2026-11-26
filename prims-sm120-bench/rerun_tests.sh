#!/usr/bin/env bash
# Steps 1-2 rerun on the current branch HEAD: CPU gate, SM120 GPU test, measured errors.
source ~/code/prims-sm120-bench/env.sh
L=~/code/prims-sm120-bench/logs
T=test/registered/kernels/ops/attention/test_flashinfer_sm120_fp8_ragged_prefill.py
echo "HEAD $(git rev-parse --short=10 HEAD)"
echo "== CPU gate"
python test/registered/unit/layers/attention/test_flashinfer_sm120_fp8_prefill_gate.py -v > $L/rerun_step1_gate.log 2>&1
grep -E "^(ok|FAIL|ERROR)|^Ran|^OK|^FAILED|^test_" $L/rerun_step1_gate.log | sed 's/ \.\.\. \[CI Test Method\].*//'
echo "== GPU test"
python $T -v > $L/rerun_step2_gpu_test.log 2>&1
grep -E "^(ok|FAIL|ERROR)|^Ran|^OK|^FAILED|^test_|Error:|Greatest" $L/rerun_step2_gpu_test.log | grep -v nvcc | sed 's/ \.\.\. \[CI Test Method\].*//' | cut -c1-250
echo "== error report"
python ~/code/prims-sm120-bench/pr1_error_report.py $T 2>&1 | grep -vE "nvcc|^\[[0-9]+/" > $L/rerun_step2_error_report.log
grep -v "^JSON" $L/rerun_step2_error_report.log
