"""Render bench_results.json as markdown tables (one per head_dim)."""

import json
import sys

rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "bench_results.json"))
ARMS = [("a_fa2", "a fa2 bf16"), ("a_auto", "a' auto bf16"), ("b", "b prims fp8"),
        ("b+q", "b + unfused quant"), ("b+fq", "b + quant (static_quant_fp8)"),
        ("b_pt", "b_pt prims fp8 (amax scales)"), ("b_pt+q", "b_pt + unfused quant"),
        ("b_pt+fq", "b_pt + quant (static_quant_fp8)")]

for D in sorted({r["head_dim"] for r in rows}):
    sub = [r for r in rows if r["head_dim"] == D]
    print(f"\n#### head_dim {D} ({sub[0]['Hq']} Q heads / {sub[0]['Hkv']} KV heads), a' picked: "
          + ", ".join(sorted({str(r.get('auto_backend')) for r in sub})))
    print("\nTime (ms, median of 50) / TFLOPS\n")
    print("| shape | " + " | ".join(n for _, n in ARMS) + " | b speedup vs a' |")
    print("|---" * (len(ARMS) + 2) + "|")
    for r in sub:
        cells = []
        for a, _ in ARMS:
            x = r.get(a, {})
            cells.append("ERR" if "error" in x else f"{x['time_ms']:.3f} / {x['tflops']:.0f}")
        def ratio(arm):
            x = r.get(arm, {"error": 1})
            return "n/a" if "error" in x or "error" in r.get("a_auto", {"error": 1}) else f"{r['a_auto']['time_ms'] / x['time_ms']:.2f}x"
        sp = f"{ratio('b')} (unfused quant {ratio('b+q')}, static_quant_fp8 {ratio('b+fq')})"
        print(f"| {r['shape']} | " + " | ".join(cells) + f" | {sp} |")
    print("\nError vs fp32 reference: cosine / max_abs / mean_abs\n")
    print("| shape | " + " | ".join(n for a, n in ARMS if "+q" not in a) + " |")
    print("|---" * (len([a for a, _ in ARMS if '+q' not in a]) + 1) + "|")
    for r in sub:
        cells = []
        for a, _ in ARMS:
            if "+q" in a:
                continue
            x = r.get(a, {})
            cells.append("ERR" if "error" in x else f"{x['cosine']:.5f} / {x['max_abs']:.3g} / {x['mean_abs']:.3g}")
        print(f"| {r['shape']} | " + " | ".join(cells) + " |")
    if any("b_quant_only_ms" in r for r in sub):
        print("\nQ/K/V quantization alone (ms): unfused / static_quant_fp8\n")
        print("| shape | b (s=1) | b_pt (amax scales) |\n|---|---|---|")
        for r in sub:
            c = [f"{r[t]['unfused']:.3f} / {r[t]['static_quant_fp8']:.3f}" if t in r else "n/a"
                 for t in ("b_quant_only_ms", "b_pt_quant_only_ms")]
            print(f"| {r['shape']} | " + " | ".join(c) + " |")
    errs = [(r["shape"], a, r[a]["error"]) for r in sub for a, _ in ARMS if "error" in r.get(a, {})]
    for s, a, e in errs:
        print(f"\n- {s} {a}: `{e}`")
