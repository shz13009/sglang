"""Print the measured error numbers for the PR 1 SM120 FP8 ragged prefill cases.

Reuses the test's own _run() (c57851364a: returns bf16_o, floor_o, floor_lse, o,
o_with_lse, lse) so inputs, plans and layer scales are identical to the test.
"floor" = FA2 on the FP8-rounded Q/K/V; "bf16" = FA2 on the bf16 Q/K/V.
Does not assert anything.
"""

import importlib.util
import json
import re
import sys

import torch

TEST_PATH = sys.argv[1]
spec = importlib.util.spec_from_file_location("pr1_test", TEST_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Pull each case's kwargs straight from the test source so they can't drift.
src = open(TEST_PATH).read()
CASES = {}
for name in re.findall(r"def (test_gqa_\w+)\(self\):", src):
    body = src.split(f"def {name}(self):", 1)[1].split("def ", 1)[0]
    kw = body.split("_assert_close_to_reference(", 1)[1].rsplit(")", 1)[0]
    kw = re.sub(r"#.*", "", kw)
    CASES[name] = eval(f"dict({kw})")


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def maxabs(a, b):
    return (a.float() - b.float()).abs().max().item()


tc = mod.TestSM120FP8RaggedPrefill()
results = {}
for name, case in CASES.items():
    bf16_o, floor_o, floor_lse, o, o_with_lse, lse = tc._run(**case)
    r = {
        "case": {k: (list(v) if isinstance(v, tuple) else v) for k, v in case.items()},
        # what the test asserts
        "prims_vs_floor_cosine": cos(o, floor_o),             # > 0.9995
        "prims_vs_floor_lse_max_abs": maxabs(lse, floor_lse),  # atol 1e-2
        "o_vs_o_with_lse_max_abs": maxabs(o, o_with_lse),      # assert_close
        "prims_vs_bf16_cosine": cos(o, bf16_o),               # > 0.99
        # extra context
        "prims_vs_floor_out_max_abs": maxabs(o, floor_o),
        "prims_vs_bf16_out_max_abs": maxabs(o, bf16_o),
        "floor_vs_bf16_cosine": cos(floor_o, bf16_o),
        "nan_in_out": bool(torch.isnan(o.float()).any()),
    }
    results[name] = r
    print(name, json.dumps(r, indent=2))

print("JSON:", json.dumps(results))
