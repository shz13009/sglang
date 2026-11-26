"""SM120 FP8 ragged prefill matches the BF16 ragged wrapper it replaces.

The output must stay within FP8 quantization error of FA2 for a ragged batch, and
the returned LSE must be in FA2's base so the prefix merge stays correct. Each
layer's softmax scale and KV scale are applied at run time, not baked into the plan.
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

        reference = BatchPrefillWithRaggedKVCacheWrapper(
            workspace, "NHD", backend="fa2"
        )
        reference.plan(
            qo_indptr,
            qo_indptr,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            causal=True,
            sm_scale=scaling,
            q_data_type=torch.bfloat16,
        )
        ref_o, ref_lse = reference.run_return_lse(q, k, v)

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
        return ref_o, ref_lse, o, o_with_lse, lse

    def _assert_close_to_reference(self, **case):
        ref_o, ref_lse, o, o_with_lse, lse = self._run(**case)
        cos = torch.nn.functional.cosine_similarity(
            o.float().flatten(), ref_o.float().flatten(), dim=0
        )
        # FP8 e4m3 Q/K/V; the issue reports error of the same order as FP8 KV itself.
        self.assertGreater(cos.item(), 0.998)
        # The LSE variant is a separate kernel specialization of the same math.
        torch.testing.assert_close(o_with_lse, o)
        # LSE feeds merge_state with the paged prefix, so it must be log2-based like FA2's.
        torch.testing.assert_close(lse, ref_lse, atol=1e-1, rtol=0)

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
            scaling=0.0625,
            kv_scale=4.0,
        )


if __name__ == "__main__":
    unittest.main()
