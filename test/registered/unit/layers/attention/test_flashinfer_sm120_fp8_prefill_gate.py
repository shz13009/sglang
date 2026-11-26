"""The SM120 FP8 ragged prefill gate refuses every configuration outside the kernel contract.

An opt-in that silently runs on an unsupported model would quantize attention it
cannot compute; each disqualifier must produce a reason, and the FP8 cast must
saturate like the KV pool instead of turning out-of-range values into NaN.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.flashinfer_sm120_fp8_prefill import (
    SM120FP8RaggedPrefill,
    quantize_to_fp8,
    sm120_fp8_ragged_prefill_unsupported_reason,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_SUPPORTED = dict(
    device_is_sm120=True,
    kv_cache_dtype=torch.float8_e4m3fn,
    model_dtype=torch.bfloat16,
    head_dim=128,
    v_head_dim=128,
    num_qo_heads=32,
    num_kv_heads=8,
    has_wrapper_dispatch=False,
    ragged_prefill_disabled=False,
)


class TestSM120FP8RaggedPrefillGate(CustomTestCase):
    def test_supported_configuration_has_no_reason(self):
        self.assertIsNone(sm120_fp8_ragged_prefill_unsupported_reason(**_SUPPORTED))

    def test_each_disqualifier_is_reported(self):
        disqualifiers = {
            "not_sm120": dict(device_is_sm120=False),
            "e5m2_kv": dict(kv_cache_dtype=torch.float8_e5m2),
            "bf16_kv": dict(kv_cache_dtype=torch.bfloat16),
            "fp32_model": dict(model_dtype=torch.float32),
            "head_dim_96": dict(head_dim=96, v_head_dim=96),
            "asymmetric_head_dims": dict(head_dim=192, v_head_dim=128),
            "uneven_gqa": dict(num_qo_heads=12, num_kv_heads=8),
            "wrapper_dispatch": dict(has_wrapper_dispatch=True),
            "ragged_disabled": dict(ragged_prefill_disabled=True),
        }
        for name, change in disqualifiers.items():
            with self.subTest(name):
                reason = sm120_fp8_ragged_prefill_unsupported_reason(
                    **{**_SUPPORTED, **change}
                )
                self.assertIsInstance(reason, str)


class TestSM120FP8RaggedPrefillLayerGate(CustomTestCase):
    """A layer outside the kernel contract keeps the regular ragged wrapper."""

    def _gate(self):
        return SM120FP8RaggedPrefill(
            wrapper=None,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=128,
            out_dtype=torch.bfloat16,
        )

    def _layer(self, **change):
        fields = dict(
            attn_type=AttentionType.DECODER,
            is_cross_attention=False,
            sliding_window_size=-1,
            logit_cap=0.0,
            tp_q_head_num=8,
            tp_k_head_num=2,
            head_dim=128,
            v_head_dim=128,
        )
        return SimpleNamespace(**{**fields, **change})

    def test_matching_decoder_layer_is_supported(self):
        self.assertTrue(self._gate().supports_layer(self._layer()))

    def test_each_unsupported_layer_falls_back(self):
        unsupported = {
            "sliding_window": dict(sliding_window_size=4096),
            "soft_cap": dict(logit_cap=30.0),
            "cross_attention": dict(is_cross_attention=True),
            "encoder_only": dict(attn_type=AttentionType.ENCODER_ONLY),
            "bidirectional": dict(attn_type=AttentionType.DECODER_BIDIRECTIONAL),
            "fewer_q_heads_than_planned": dict(tp_q_head_num=4),
            "other_kv_heads": dict(tp_k_head_num=1),
            "other_head_dim": dict(head_dim=64, v_head_dim=64),
        }
        gate = self._gate()
        for name, change in unsupported.items():
            with self.subTest(name):
                self.assertFalse(gate.supports_layer(self._layer(**change)))


class TestQuantizeToFP8(CustomTestCase):
    def test_out_of_range_values_saturate_and_scale_divides(self):
        finfo = torch.finfo(torch.float8_e4m3fn)
        x = torch.tensor([[1000.0, -1000.0, 2.0, 0.5]], dtype=torch.bfloat16)

        out = quantize_to_fp8(x, scale=None).float()
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(out[0, 0].item(), finfo.max)
        self.assertEqual(out[0, 1].item(), -finfo.max)

        scaled = quantize_to_fp8(x, scale=4.0).float()
        self.assertEqual(scaled[0, 2].item(), 0.5)
        self.assertEqual(scaled[0, 3].item(), 0.125)

    def test_output_is_contiguous_for_strided_input(self):
        x = torch.randn(8, 4, 16, dtype=torch.bfloat16).transpose(0, 1)
        self.assertTrue(quantize_to_fp8(x, scale=None).is_contiguous())


if __name__ == "__main__":
    unittest.main()
