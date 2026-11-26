"""SM120 ragged causal prefill microbenchmark: fa2 bf16 vs auto bf16 vs cute-dsl-prims fp8.

Arms (all NHD, ragged, qo_indptr == kv_indptr, causal, no window / soft-cap):
  a      backend="fa2",            bf16 Q/K/V, sm_scale = D**-0.5
  a'     backend="auto",           bf16 Q/K/V (records the backend FlashInfer picks)
  b      backend="cute-dsl-prims", fp8_e4m3 Q/K/V, s=1.0, planned as in PR 1
  b_pt   same as b, per-tensor scales s = amax / 448
  b+q / b_pt+q  same kernels with the Q/K/V quantization inside the timed region
Errors are against an fp32 causal attention on the same bf16 inputs.
"""

import json
import statistics
import sys

import torch
from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

FP8 = torch.float8_e4m3fn
WS = 256 * 1024 * 1024
WARMUP, ITERS = 10, 50

HEAD_CFGS = [(128, 32, 8), (256, 24, 4)]  # (D, Hq, Hkv)
SHAPES = {"1x4k": [4096], "1x16k": [16384], "1x32k": [32768], "4x4k": [4096] * 4}


def quant(x, s):
    return (x.float() / s).clamp(-448, 448).to(FP8)


def time_ms(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


@torch.no_grad()
def ref_fp32(q, k, v, seq_lens, D, Hq, Hkv, blk=2048):
    """Chunked fp32 causal attention; q: [T,Hq,D], k/v: [T,Hkv,D] (bf16 in)."""
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    g = Hq // Hkv
    scale = D**-0.5
    off = 0
    for L in seq_lens:
        qs = q[off:off + L].float().transpose(0, 1)  # [Hq,L,D]
        ks = k[off:off + L].float().transpose(0, 1)
        vs = v[off:off + L].float().transpose(0, 1)
        for h in range(Hq):
            kh, vh = ks[h // g], vs[h // g]
            for s0 in range(0, L, blk):
                s1 = min(L, s0 + blk)
                sc = (qs[h, s0:s1] @ kh[:s1].T) * scale  # [b, s1]
                rows = torch.arange(s0, s1, device=q.device)[:, None]
                cols = torch.arange(s1, device=q.device)[None, :]
                sc.masked_fill_(cols > rows, float("-inf"))
                out[off + s0:off + s1, h] = torch.softmax(sc, dim=-1) @ vh[:s1]
        off += L
    return out


def errs(o, ref):
    of = o.float()
    d = (of - ref).abs()
    return {
        "cosine": torch.nn.functional.cosine_similarity(of.flatten(), ref.flatten(), dim=0).item(),
        "max_abs": d.max().item(),
        "mean_abs": d.mean().item(),
        "rel_l2": ((of - ref).norm() / ref.norm()).item(),
    }


try:
    from sglang.srt.layers.attention.flashinfer_sm120_fp8_prefill import (
        quantize_to_fp8 as branch_quant,
    )
except Exception as _ex:  # bench still runs without the branch
    print("branch quantize_to_fp8 unavailable:", _ex)
    branch_quant = None


def main():
    dev = torch.device("cuda")
    print("torch", torch.__version__, "cuda", torch.version.cuda,
          "cap", torch.cuda.get_device_capability(), torch.cuda.get_device_name())
    import flashinfer
    print("flashinfer", flashinfer.__version__)
    ws = torch.empty(WS, dtype=torch.uint8, device=dev)
    rows = []
    for D, Hq, Hkv in HEAD_CFGS:
        for shape_name, seq_lens in SHAPES.items():
            torch.manual_seed(0)
            T = sum(seq_lens)
            q = torch.randn(T, Hq, D, device=dev, dtype=torch.bfloat16)
            k = torch.randn(T, Hkv, D, device=dev, dtype=torch.bfloat16)
            v = torch.randn(T, Hkv, D, device=dev, dtype=torch.bfloat16)
            indptr = torch.tensor([0] + torch.tensor(seq_lens).cumsum(0).tolist(),
                                  dtype=torch.int32, device=dev)
            flops = 4 * sum(L * L / 2 for L in seq_lens) * Hq * D
            ref = ref_fp32(q, k, v, seq_lens, D, Hq, Hkv)
            row = {"head_dim": D, "Hq": Hq, "Hkv": Hkv, "shape": shape_name}

            def record(arm, fn, out=None):
                try:
                    o = fn() if out is None else out
                    t = time_ms(fn)
                    e = errs(o, ref)
                    row[arm] = {"time_ms": t, "tflops": flops / t / 1e9, **e}
                except Exception as ex:  # report, keep going
                    row[arm] = {"error": f"{type(ex).__name__}: {ex}"[:400]}
                print(D, shape_name, arm, json.dumps(row[arm]), flush=True)

            # a / a' : bf16
            for arm, backend in (("a_fa2", "fa2"), ("a_auto", "auto")):
                try:
                    w = BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend=backend)
                    w.plan(indptr, indptr, Hq, Hkv, D, causal=True, sm_scale=D**-0.5,
                           q_data_type=torch.bfloat16)
                    if backend == "auto":
                        row["auto_backend"] = str(getattr(w, "_backend", "?"))
                    record(arm, lambda w=w: w.run(q, k, v))
                except Exception as ex:
                    row[arm] = {"error": f"{type(ex).__name__}: {ex}"[:400]}
                    print(D, shape_name, arm, row[arm], flush=True)

            # b : fp8 prims, planned like PR 1
            try:
                w8 = BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="cute-dsl-prims")
                w8.plan(indptr, indptr, Hq, Hkv, D, causal=True, sm_scale=1.0,
                        q_data_type=FP8, kv_data_type=FP8, o_data_type=torch.bfloat16)
                for tag, (qs, ks, vs) in (
                    ("b", (1.0, 1.0, 1.0)),
                    ("b_pt", tuple(x.abs().amax().float().item() / 448 for x in (q, k, v))),
                ):
                    q8, k8, v8 = quant(q, qs), quant(k, ks), quant(v, vs)
                    kw = dict(q_scale=D**-0.5 * qs, k_scale=ks, v_scale=vs)
                    record(tag, lambda q8=q8, k8=k8, v8=v8, kw=kw: w8.run(q8, k8, v8, **kw))
                    record(tag + "+q", lambda qs=qs, ks=ks, vs=vs, kw=kw: w8.run(
                        quant(q, qs), quant(k, ks), quant(v, vs), **kw))
                    if branch_quant is not None:
                        # The branch's quantize_to_fp8 (static_quant_fp8 Triton kernel), as
                        # SM120FP8RaggedPrefill.forward calls it: Q scale None when s == 1.
                        bq = None if tag == "b" else qs
                        record(tag + "+fq", lambda bq=bq, ks=ks, vs=vs, kw=kw: w8.run(
                            branch_quant(q, bq), branch_quant(k, ks), branch_quant(v, vs), **kw))
                        row[tag + "_quant_only_ms"] = {
                            "unfused": time_ms(lambda qs=qs, ks=ks, vs=vs: (quant(q, qs), quant(k, ks), quant(v, vs))),
                            "static_quant_fp8": time_ms(lambda bq=bq, ks=ks, vs=vs: (
                                branch_quant(q, bq), branch_quant(k, ks), branch_quant(v, vs))),
                        }
                        print(D, shape_name, tag, "quant_only_ms", row[tag + "_quant_only_ms"], flush=True)
                    row[tag]["scales"] = [qs, ks, vs] if "error" not in row[tag] else None
            except Exception as ex:
                row["b"] = {"error": f"{type(ex).__name__}: {ex}"[:400]}
                print(D, shape_name, "b", row["b"], flush=True)

            rows.append(row)
            del ref, q, k, v
            torch.cuda.empty_cache()

    out = sys.argv[1] if len(sys.argv) > 1 else "bench_results.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
