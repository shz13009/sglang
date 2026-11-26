"""Out-of-tree probe: static_quant_fp8 on [T, H, D] (as the branch calls it) vs a [T, H*D] view.

static_quant_fp8 launches one Triton program per row of the last dim, so [T, H, D]
gives rows of D elements (BLOCK=D, 1 warp). The [T, H*D] view is the same bytes.
"""

import statistics

import torch

from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8


def time_ms(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


s = torch.tensor([1.0], dtype=torch.float32, device="cuda")
print("| head_dim | tokens | [T,H,D] ms | [T,H*D] ms | speedup | GB/s [T,H,D] -> [T,H*D] | bit-identical |")
print("|---|---|---|---|---|---|---|")
for D, Hq, Hkv in ((128, 32, 8), (256, 24, 4)):
    for T in (4096, 16384, 32768):
        torch.manual_seed(0)
        q = torch.randn(T, Hq, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(T, Hkv, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(T, Hkv, D, device="cuda", dtype=torch.bfloat16)
        three_d = lambda: [static_quant_fp8(x, s)[0] for x in (q, k, v)]
        two_d = lambda: [static_quant_fp8(x.view(T, -1), s)[0].view(x.shape) for x in (q, k, v)]
        same = all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(three_d(), two_d()))
        t3, t2 = time_ms(three_d), time_ms(two_d)
        nbytes = sum(x.numel() for x in (q, k, v)) * 3  # 2 B read + 1 B write per element
        print(f"| {D} | {T} | {t3:.3f} | {t2:.3f} | {t3 / t2:.2f}x | "
              f"{nbytes / t3 / 1e6:.0f} -> {nbytes / t2 / 1e6:.0f} | {same} |")
