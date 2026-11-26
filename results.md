# #52389 PR 1 on RTX 5090 (SM120): test results and baseline

Tested commit: `c57851364a` on `feat/flashinfer-sm120-fp8-ragged-prefill` (steps 1–3 were also run on `b2fef73c08`; see the history below).
No sglang source or test files were changed for these runs. The scripts, raw JSON and logs are in [`prims-sm120-bench/`](prims-sm120-bench/).

## Summary

| step | result on c57851364a |
|---|---|
| 1. CPU gate unit test | **PASS** 4/4 |
| 2. SM120 GPU test | **PASS** 3/3. Cosine vs FA2 on FP8 inputs: 0.99984 (hd128), 0.99995 (hd256). LSE max abs 0.0023 / 0.0020 |
| 3. Dense FlashInfer regression (switch unset) | **PASS** 8/8 (also 8/8 on b2fef73c08) |
| 4a. Server with switch on | starts; log shows `FlashInfer prefill: SM120 FP8 ragged attention enabled.` |
| 4b. GSM8K, 500 q, 5-shot, T=0 | off 0.924 / on 0.924 (invalid 0.000 / 0.002) |
| 4c. Prefill-heavy TTFT / throughput | on/off between 0.95 and 1.08: no gain beyond noise (see "Why the end-to-end gain is small") |
| 4d. Kernel microbench, 4×4k row vs `auto` (= fa2) | prims fp8 + `static_quant_fp8`: **1.27× (hd128), 1.66× (hd256)**. Kernel alone 1.94× / 2.03× |

Suggested follow-ups (for the Mac session to decide):
1. **Flatten the input to `quantize_to_fp8` before calling `static_quant_fp8`.** The Triton kernel runs one program per row of the last dim, so `[T, H, D]` gives 128-element rows with 1 warp. With a `[T, H*D]` view the quantization is 3.3–4.7× faster (hd128) and 1.8–3.1× faster (hd256), with bit-identical output, at about 1.3–1.8 TB/s. Projected 4×4k vs `auto`: about 1.65× (hd128) and 1.76× (hd256); this is not measured through the branch.
   ```python
   def quantize_to_fp8(x, scale):
       x = x.contiguous()
       scale_tensor = _scale_tensor(1.0 if scale is None else scale, x.device)
       return static_quant_fp8(x.view(x.shape[0], -1), scale_tensor)[0].view(x.shape)
   ```
2. End-to-end, with `chunked_prefill_size=4096` (this GPU's default) the ragged part is at most 4096² per chunk. The paged-prefix attention, which is most of the attention at 32k, still runs the existing path. Measuring with a larger `--chunked-prefill-size`, or moving the paged-prefix part to FP8, is where an end-to-end gain would show.

## Environment

- Host OMEN, Windows 11 + WSL2 Ubuntu (kernel 6.18.33.2-microsoft-standard-WSL2). `.wslconfig`: memory=24GB, swap=24GB, unchanged.
- RTX 5090 32 GB, compute capability 12.0. Driver 620.67.01 (CUDA UMD 13.5). CUDA toolkit 13.3 in WSL. The GPU also drives the Windows desktop (about 2 GB used at idle); Ollama had no models loaded during the runs.
- Python 3.12.14 venv (uv). `SGLANG_BUILD_RUST_EXTS=none uv pip install -e "python[test]"`. torch 2.14.1+cu130 (arch list includes sm_120), flashinfer_python 0.7.4.
- Two environment workarounds, neither a repo change:
  - cargo is not installed, so the Rust extensions were skipped (`SGLANG_BUILD_RUST_EXTS=none`, as in the repo's own cookbook).
  - `~/.local/bin/gcc` is an nvim-tools gcc wrapper with no `cc1plus`, and it shadows `/usr/bin/g++`. FlashInfer's JIT failed with `cannot execute 'cc1plus'`. All runs source `prims-sm120-bench/env.sh`, which puts `/usr/bin` first on PATH.

## Step 2 detail: measured errors (`pr1_error_report.py`, reuses the test's `_run`)

"floor" = FA2 on the FP8-rounded Q/K/V; "bf16" = FA2 on the bf16 Q/K/V.

| case | prims vs floor cos (> 0.9995) | prims vs floor LSE max abs (atol 1e-2) | o vs o_with_lse | prims vs bf16 cos (> 0.99) | prims vs floor out max abs | prims vs bf16 out max abs | floor vs bf16 cos |
|---|---|---|---|---|---|---|---|
| hd128, 32/8 heads, seq (1,17,300,1024), scale D^-0.5, kv_scale 1 | 0.999841 | 0.00233 | 0.0 | 0.999026 | 0.0547 | 0.172 | 0.999179 |
| hd256, 24/4 heads, seq (5,513,2048), scale 0.05, kv_scale 4 | 0.999952 | 0.00204 | 0.0 | 0.996885 | 0.344 | 2.84 | 0.996930 |

History: on `b2fef73c08` both cases failed, because the test compared against bf16 inputs with cos > 0.998 and LSE atol 0.1. Measured there: hd128 cos 0.99903 with LSE max abs 0.137 (12 of 42,944 elements over 0.1); hd256 (scaling 0.0625) cos 0.99632 with LSE max abs 0.880. FA2 on FP8-rounded inputs gave the same numbers (cos 0.99918 / 0.99635, LSE 0.137 / 0.880), so the FP8 rounding of the inputs, not the kernel, accounted for the gap. c57851364a's test reflects that.

## Kernel microbenchmark (`ragged_prefill_bench.py`)

NHD ragged, qo_indptr == kv_indptr, causal, randn bf16 inputs (seed 0). 10 warmup iterations, then the median of 50 CUDA-event timings, with plans built outside the timed region. FLOPs = 4·Σ(L²/2)·Hq·D. Arms:
- **a:** `fa2`, bf16.
- **a':** `auto`, bf16. `_backend` after plan is **fa2** for every shape; no fmha_v2.
- **b:** `cute-dsl-prims`, fp8, planned as in PR 1 (sm_scale=1.0, q_scale = D^-0.5·s); s = 1.
- **b_pt:** as b, with per-tensor scales s = amax/448.
- **"+ unfused quant":** `(x/s).clamp(±448).to(fp8)` in the timed region.
- **"+ quant (static_quant_fp8)":** the branch's `quantize_to_fp8` in the timed region.

Errors are against fp32 causal attention on the same bf16 inputs. Run on c57851364a (`bench_results_c5785136.json`; the earlier b2fef73c08 run is `bench_results.json`).

#### head_dim 128 (32 Q heads / 8 KV heads), a' picked: fa2

Time (ms, median of 50) / TFLOPS

| shape | a fa2 bf16 | a' auto bf16 | b prims fp8 | b + unfused quant | b + quant (static_quant_fp8) | b_pt prims fp8 (amax scales) | b_pt + unfused quant | b_pt + quant (static_quant_fp8) | b speedup vs a' |
|---|---|---|---|---|---|---|---|---|---|
| 1x4k | 0.838 / 164 | 0.844 / 163 | 0.388 / 354 | 0.601 / 228 | 0.569 / 242 | 0.388 / 354 | 0.604 / 228 | 0.573 / 240 | 2.17x (unfused quant 1.40x, static_quant_fp8 1.48x) |
| 1x16k | 10.281 / 214 | 10.278 / 214 | 4.981 / 441 | 6.379 / 345 | 5.639 / 390 | 4.971 / 442 | 6.387 / 344 | 5.629 / 391 | 2.06x (unfused quant 1.61x, static_quant_fp8 1.82x) |
| 1x32k | 38.938 / 226 | 38.843 / 226 | 19.720 / 446 | 22.867 / 385 | 21.038 / 418 | 19.716 / 446 | 22.940 / 383 | 21.049 / 418 | 1.97x (unfused quant 1.70x, static_quant_fp8 1.85x) |
| 4x4k | 2.653 / 207 | 2.652 / 207 | 1.370 / 401 | 2.825 / 195 | 2.093 / 263 | 1.358 / 405 | 2.816 / 195 | 2.092 / 263 | 1.94x (unfused quant 0.94x, static_quant_fp8 1.27x) |

Error vs fp32 reference: cosine / max_abs / mean_abs

| shape | a fa2 bf16 | a' auto bf16 | b prims fp8 | b + quant (static_quant_fp8) | b_pt prims fp8 (amax scales) | b_pt + quant (static_quant_fp8) |
|---|---|---|---|---|---|---|
| 1x4k | 1.00000 / 0.00858 / 8.31e-05 | 1.00000 / 0.00858 / 8.31e-05 | 0.99884 / 0.177 / 0.00203 | 0.99884 / 0.177 / 0.00203 | 0.99885 / 0.17 / 0.00202 | 0.99885 / 0.17 / 0.00202 |
| 1x16k | 1.00000 / 0.00858 / 4.35e-05 | 1.00000 / 0.00858 / 4.35e-05 | 0.99879 / 0.237 / 0.00106 | 0.99879 / 0.237 / 0.00106 | 0.99880 / 0.156 / 0.00105 | 0.99880 / 0.156 / 0.00105 |
| 1x32k | 1.00000 / 0.00845 / 3.13e-05 | 1.00000 / 0.00845 / 3.13e-05 | 0.99878 / 0.161 / 0.000757 | 0.99878 / 0.161 / 0.000757 | 0.99879 / 0.172 / 0.000752 | 0.99879 / 0.172 / 0.000752 |
| 4x4k | 1.00000 / 0.00925 / 8.27e-05 | 1.00000 / 0.00925 / 8.27e-05 | 0.99883 / 0.237 / 0.00202 | 0.99883 / 0.237 / 0.00202 | 0.99885 / 0.217 / 0.00201 | 0.99885 / 0.217 / 0.00201 |

Q/K/V quantization alone (ms): unfused / static_quant_fp8

| shape | b (s=1) | b_pt (amax scales) |
|---|---|---|
| 1x4k | 0.266 / 0.242 | 0.276 / 0.241 |
| 1x16k | 1.548 / 0.791 | 1.541 / 0.791 |
| 1x32k | 4.249 / 1.525 | 3.494 / 1.522 |
| 4x4k | 1.542 / 0.787 | 1.533 / 0.787 |

#### head_dim 256 (24 Q heads / 4 KV heads), a' picked: fa2

Time (ms, median of 50) / TFLOPS

| shape | a fa2 bf16 | a' auto bf16 | b prims fp8 | b + unfused quant | b + quant (static_quant_fp8) | b_pt prims fp8 (amax scales) | b_pt + unfused quant | b_pt + quant (static_quant_fp8) | b speedup vs a' |
|---|---|---|---|---|---|---|---|---|---|
| 1x4k | 1.200 / 172 | 1.213 / 170 | 0.562 / 367 | 0.976 / 211 | 0.673 / 306 | 0.559 / 369 | 0.976 / 211 | 0.675 / 305 | 2.16x (unfused quant 1.24x, static_quant_fp8 1.80x) |
| 1x16k | 15.716 / 210 | 15.714 / 210 | 7.327 / 450 | 9.266 / 356 | 7.766 / 425 | 7.305 / 452 | 9.301 / 355 | 7.763 / 425 | 2.14x (unfused quant 1.70x, static_quant_fp8 2.02x) |
| 1x32k | 62.069 / 213 | 62.115 / 212 | 28.861 / 457 | 33.141 / 398 | 29.766 / 443 | 28.885 / 457 | 33.163 / 398 | 29.752 / 443 | 2.15x (unfused quant 1.87x, static_quant_fp8 2.09x) |
| 4x4k | 4.097 / 201 | 4.101 / 201 | 2.025 / 407 | 4.062 / 203 | 2.469 / 334 | 1.996 / 413 | 4.067 / 203 | 2.465 / 335 | 2.03x (unfused quant 1.01x, static_quant_fp8 1.66x) |

Error vs fp32 reference: cosine / max_abs / mean_abs

| shape | a fa2 bf16 | a' auto bf16 | b prims fp8 | b + quant (static_quant_fp8) | b_pt prims fp8 (amax scales) | b_pt + quant (static_quant_fp8) |
|---|---|---|---|---|---|---|
| 1x4k | 1.00000 / 0.0084 / 8.23e-05 | 1.00000 / 0.0084 / 8.23e-05 | 0.99885 / 0.168 / 0.00201 | 0.99885 / 0.168 / 0.00201 | 0.99885 / 0.172 / 0.00201 | 0.99885 / 0.172 / 0.00201 |
| 1x16k | 1.00000 / 0.00932 / 4.33e-05 | 1.00000 / 0.00932 / 4.33e-05 | 0.99879 / 0.216 / 0.00105 | 0.99879 / 0.216 / 0.00105 | 0.99881 / 0.181 / 0.00104 | 0.99881 / 0.181 / 0.00104 |
| 1x32k | 1.00000 / 0.00884 / 3.13e-05 | 1.00000 / 0.00884 / 3.13e-05 | 0.99880 / 0.179 / 0.000752 | 0.99880 / 0.179 / 0.000752 | 0.99881 / 0.194 / 0.000748 | 0.99881 / 0.194 / 0.000748 |
| 4x4k | 1.00000 / 0.00932 / 8.26e-05 | 1.00000 / 0.00932 / 8.26e-05 | 0.99886 / 0.216 / 0.00201 | 0.99886 / 0.216 / 0.00201 | 0.99887 / 0.209 / 0.002 | 0.99887 / 0.209 / 0.002 |

Q/K/V quantization alone (ms): unfused / static_quant_fp8

| shape | b (s=1) | b_pt (amax scales) |
|---|---|---|
| 1x4k | 0.501 / 0.172 | 0.501 / 0.181 |
| 1x16k | 2.148 / 0.545 | 2.152 / 0.548 |
| 1x32k | 4.690 / 1.042 | 4.679 / 1.042 |
| 4x4k | 2.147 / 0.549 | 2.138 / 0.548 |

### `static_quant_fp8` layout probe (`quant_layout_probe.py`, outside the source tree)

The Q/K/V quantization was timed both as the branch calls it (`[T, H, D]`) and on a flattened `[T, H*D]` view of the same tensors. Scale is 1.0, and outputs are compared byte for byte.

| head_dim | tokens | [T,H,D] ms | [T,H*D] ms | speedup | GB/s [T,H,D] -> [T,H*D] | bit-identical |
|---|---|---|---|---|---|---|
| 128 | 4096 | 0.239 | 0.051 | 4.65x | 316 -> 1467 | True |
| 128 | 16384 | 0.790 | 0.239 | 3.31x | 382 -> 1263 | True |
| 128 | 32768 | 1.524 | 0.431 | 3.54x | 396 -> 1402 | True |
| 256 | 4096 | 0.168 | 0.055 | 3.08x | 598 -> 1842 | True |
| 256 | 16384 | 0.546 | 0.306 | 1.79x | 738 -> 1318 | True |
| 256 | 32768 | 1.032 | 0.557 | 1.85x | 780 -> 1445 | True |

## End-to-end smoke (`e2e.sh`)

Settings: Qwen/Qwen3-8B, `--attention-backend flashinfer --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 --random-seed 0`.
- Server-chosen limits: chunked_prefill_size=4096, max_prefill_tokens=16384, max_total_num_tokens=149609, context_len=40960.
- Both runs on c57851364a. "off" leaves `SGLANG_ENABLE_FLASHINFER_SM120_FP8_RAGGED_PREFILL` unset; "on" sets it to 1.
- GSM8K: `python -m sglang.test.few_shot_gsm8k --num-questions 500 --num-shots 5 --parallel 64 --temperature 0`.
- Prefill-heavy: `python -m sglang.benchmark.serving --dataset-name random-ids --random-output-len 1 --random-range-ratio 1.0`.
  - Per input length: one discarded 2-prompt warmup, then C=1 with 8 prompts and C=4 with 16 prompts.
  - `/flush_cache` before every run, and a distinct seed for each run.

| run | GSM8K acc (500 q, 5-shot, T=0) | invalid | GSM8K latency (s) |
|---|---|---|---|
| off | 0.924 | 0.0 | 26.295 |
| on | 0.924 | 0.002 | 24.536 |

| input len (target) | C | prompts | actual mean input tok | off median TTFT (ms) | on median TTFT (ms) | on/off | off input tok/s | on input tok/s | on/off |
|---|---|---|---|---|---|---|---|---|---|
| 8192 | 1 | 8 | 8192 | 751 | 712 | 0.948 | 12121 | 13129 | 1.083 |
| 8192 | 4 | 16 | 8192 | 2788 | 2719 | 0.975 | 11902 | 11918 | 1.001 |
| 16384 | 1 | 8 | 16384 | 1732 | 1840 | 1.063 | 10626 | 10071 | 0.948 |
| 16384 | 4 | 16 | 16384 | 6930 | 6591 | 0.951 | 9992 | 10071 | 1.008 |
| 32768 | 1 | 8 | 32768 | 4277 | 4296 | 1.004 | 8519 | 8491 | 0.997 |
| 32768 | 4 | 16 | 32768 | 17025 | 16928 | 0.994 | 7905 | 8186 | 1.035 |

Caveats, which apply equally to both arms (the same seeds give the same prompts):
- `bench_serving` sends its own warmup request, which reuses the first prompt. So 1 prompt per run is served from the prefix cache. Medians are robust to this; means are slightly low.
- `random-ids` decodes random token ids to text, and the server re-tokenizes it. Server-side lengths are therefore about 0–24% above the target (for example 8,262–10,158 tokens for the 8192 target). The "actual mean input tok" column is bench_serving's own count.
- A first off run on b2fef73c08 reused seed 42 for C=1 and C=4, so the C=4 prompts were served from cache (C=4 median TTFT came out below C=1). It is discarded and not included here. GSM8K on that run was also 0.924.

### Why the end-to-end gain is small

With chunked_prefill_size=4096, each prefill chunk's ragged attention is at most 4096×4096 causal. For Qwen3-8B that is roughly 4.9 TFLOP across 36 layers, against roughly 65 TFLOP of linear layers for the same chunk. Halving the ragged part can save only a few percent, which is the size of the noise here.

From the second chunk on, the attention over the cached prefix (up to 28k tokens at the 32k target) goes through the paged wrapper. This PR leaves that path unchanged.

The kernel-level 1.3–2.1× therefore only shows end to end once the ragged share grows (a larger `--chunked-prefill-size`) or the paged-prefix part also moves to FP8.

## Reproduce

```bash
# WSL, ~/code/sglang on the branch; env.sh activates .venv and puts /usr/bin and CUDA first on PATH
bash prims-sm120-bench/rerun_tests.sh       # steps 1-2 + measured errors
bash prims-sm120-bench/rerun_bench.sh       # step 3 + microbenchmark
python prims-sm120-bench/quant_layout_probe.py
bash prims-sm120-bench/e2e.sh off && bash prims-sm120-bench/e2e.sh on
python prims-sm120-bench/collect_e2e.py
```
(The scripts expect to live in `~/code/prims-sm120-bench/`, as they did on OMEN.)

