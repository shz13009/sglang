"""Collect GSM8K and prefill-heavy numbers from e2e/{off,on} into a markdown table."""

import re
from pathlib import Path

B = Path.home() / "code/prims-sm120-bench/e2e"


def grab(text, label):
    m = re.search(rf"{re.escape(label)}\s*:?\s*([0-9.]+)", text)
    return float(m.group(1)) if m else float("nan")


print("| run | GSM8K acc (500 q, 5-shot, T=0) | invalid | GSM8K latency (s) |")
print("|---|---|---|---|")
for mode in ("off", "on"):
    t = (B / mode / "gsm8k.log").read_text()
    print(f"| {mode} | {grab(t, 'Accuracy')} | {grab(t, 'Invalid')} | {grab(t, 'Latency')} |")

print("\n| input len (target) | C | prompts | actual mean input tok | off median TTFT (ms) | on median TTFT (ms) | on/off | "
      "off input tok/s | on input tok/s | on/off |")
print("|---|---|---|---|---|---|---|---|---|---|")
for L in (8192, 16384, 32768):
    for C in (1, 4):
        r = {}
        for mode in ("off", "on"):
            t = (B / mode / f"prefill_{L}_c{C}.log").read_text()
            r[mode] = dict(
                ttft=grab(t, "Median TTFT (ms)"),
                tput=grab(t, "Input token throughput (tok/s)"),
                n=grab(t, "Successful requests"),
                tin=grab(t, "Total input tokens"),
            )
        n = r["off"]["n"]
        print(f"| {L} | {C} | {n:.0f} | {r['off']['tin'] / n:.0f} | {r['off']['ttft']:.0f} | {r['on']['ttft']:.0f} | "
              f"{r['on']['ttft'] / r['off']['ttft']:.3f} | {r['off']['tput']:.0f} | {r['on']['tput']:.0f} | "
              f"{r['on']['tput'] / r['off']['tput']:.3f} |")
