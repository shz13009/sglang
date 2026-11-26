"""SM120 FP8 ragged prefill computes the attention FA2 computes on the same FP8 inputs.

Against FA2 run on the FP8-rounded Q/K/V, the output and the LSE must match tightly:
a wrong scale, a plan that bakes in one layer's softmax scale, or an LSE in another
base (which would corrupt the merge with the paged prefix) all break that. Against
the BF16 inputs only a loose bound holds, since FP8 rounding alone costs that much.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.utils.device import is_sm120
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")

_WORKSPACE_BYTES = 128 * 1024 * 1024


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def _layer(*, num_qo_heads, num_kv_heads, head_dim, scaling, kv_scale):
    return SimpleNamespace(
        attn_type=AttentionType.DECODER,
        is_cross_attention=False,
        sliding_window_size=-1,
        logit_cap=0.0,
        tp_q_head_num=num_qo_heads,
        tp_k_head_num=num_kv_heads,
        head_dim=head_dim,
        v_head_dim=head_dim,
        scaling=scaling,
        q_scale_float=None,
        k_scale_float=kv_scale,
        v_scale_float=kv_scale,
    )


@unittest.skipUnless(
    torch.cuda.is_available() and is_flashinfer_available() and is_sm120(),
    "an SM120 GPU (compute capability 12.0) and flashinfer are required",
)
class TestSM120FP8RaggedPrefill(CustomTestCase):
    def _run(
        self, *, num_qo_heads, num_kv_heads, head_dim, seq_lens, scaling, kv_scale
    ):
        from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

        from sglang.srt.layers.attention.flashinfer_sm120_fp8_prefill import (
            SM120FP8RaggedPrefill,
            quantize_to_fp8,
        )

        torch.manual_seed(0)
        workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")
        qo_indptr = torch.tensor(
            [0] + torch.tensor(seq_lens).cumsum(0).tolist(),
            dtype=torch.int32,
            device="cuda",
        )
        num_tokens = sum(seq_lens)
        q = torch.randn(num_tokens, num_qo_heads, head_dim, device="cuda").bfloat16()
        # K/V scaled by kv_scale so a non-unit KV scale is exercised.
        k = (
            kv_scale * torch.randn(num_tokens, num_kv_heads, head_dim, device="cuda")
        ).bfloat16()
        v = (
            kv_scale * torch.randn(num_tokens, num_kv_heads, head_dim, device="cuda")
        ).bfloat16()

        def fa2(q_in, k_in, v_in):
            wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                workspace, "NHD", backend="fa2"
            )
            wrapper.plan(
                qo_indptr,
                qo_indptr,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                causal=True,
                sm_scale=scaling,
                q_data_type=torch.bfloat16,
            )
            return wrapper.run_return_lse(q_in, k_in, v_in)

        def fp8_rounded(x, scale):
            # Exact in bf16: FP8 values are representable and kv_scale is a power of 2.
            return quantize_to_fp8(x, scale).to(torch.bfloat16) * (scale or 1.0)

        bf16_o, _ = fa2(q, k, v)
        floor_o, floor_lse = fa2(
            fp8_rounded(q, None), fp8_rounded(k, kv_scale), fp8_rounded(v, kv_scale)
        )

        fp8 = SM120FP8RaggedPrefill(
            wrapper=BatchPrefillWithRaggedKVCacheWrapper(
                workspace, "NHD", backend="cute-dsl-prims"
            ),
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            out_dtype=torch.bfloat16,
        )
        fp8.plan(qo_indptr=qo_indptr)
        layer = _layer(
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scaling=scaling,
            kv_scale=kv_scale,
        )
        self.assertTrue(fp8.supports_layer(layer))
        o = fp8.forward(q=q, k=k, v=v, layer=layer, return_lse=False)
        o_with_lse, lse = fp8.forward(q=q, k=k, v=v, layer=layer, return_lse=True)
        return bf16_o, floor_o, floor_lse, o, o_with_lse, lse

    def _assert_close_to_reference(self, **case):
        bf16_o, floor_o, floor_lse, o, o_with_lse, lse = self._run(**case)
        # Measured on an RTX 5090: 0.99984 (hd128) and 0.99996 (hd256).
        self.assertGreater(_cosine(o, floor_o), 0.9995)
        # LSE feeds merge_state with the paged prefix, so it must be log2-based like
        # FA2's; measured max abs diff 0.0023, a base mismatch is off by ~30%.
        torch.testing.assert_close(lse, floor_lse, atol=1e-2, rtol=0)
        # The LSE variant is a separate kernel specialization of the same math.
        torch.testing.assert_close(o_with_lse, o)
        # FP8 rounding alone measured 0.99918 (hd128) and 0.99635 (hd256, 4x logits).
        self.assertGreater(_cosine(o, bf16_o), 0.99)

    def test_gqa_head_dim_128_ragged_batch(self):
        self._assert_close_to_reference(
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            seq_lens=(1, 17, 300, 1024),
            scaling=128**-0.5,
            kv_scale=1.0,
        )

    def test_gqa_head_dim_256_with_layer_scaling_and_kv_scale(self):
        self._assert_close_to_reference(
            num_qo_heads=24,
            num_kv_heads=4,
            head_dim=256,
            seq_lens=(5, 513, 2048),
            # Not 256**-0.5, so a plan that keeps the default softmax scale fails.
            scaling=0.05,
            kv_scale=4.0,
        )

    def test_quantization_saturates_like_the_kv_pool(self):
        from sglang.srt.layers.attention.flashinfer_sm120_fp8_prefill import (
            quantize_to_fp8,
        )

        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        x = torch.tensor(
            [[1000.0, -1000.0, 2.0, 0.5]], dtype=torch.bfloat16, device="cuda"
        )
        out = quantize_to_fp8(x, scale=None).float()
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(out[0, :2].tolist(), [fp8_max, -fp8_max])
        self.assertEqual(
            quantize_to_fp8(x, scale=4.0).float()[0, 2:].tolist(), [0.5, 0.125]
        )
        # A strided K/V view (e.g. a slice of a fused qkv) still quantizes correctly.
        wide = torch.randn(8, 3, 4, 16, dtype=torch.bfloat16, device="cuda")
        strided = quantize_to_fp8(wide[:, 1], scale=None)
        packed = quantize_to_fp8(wide[:, 1].contiguous(), scale=None)
        self.assertTrue(
            torch.equal(strided.view(torch.uint8), packed.view(torch.uint8))
        )


if __name__ == "__main__":
    unittest.main()
