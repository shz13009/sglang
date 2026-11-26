"""SM120 FP8 FMHA for the ragged (new-token) part of FlashInfer extend.

Opt-in through ``SGLANG_ENABLE_FLASHINFER_SM120_FP8_RAGGED_PREFILL``. With an
FP8 KV cache, the new tokens of an extend batch are quantized to FP8 and run
through FlashInfer's ``cute-dsl-prims`` backend instead of the BF16/FP16 ragged
kernel. The cached-prefix part keeps the paged wrapper, and any layer outside
the kernel's contract (sliding window, soft-cap, non-causal, head-count
mismatch) keeps the regular ragged wrapper. See sgl-project/sglang#52389.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch

from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
from sglang.srt.layers.radix_attention import AttentionType

if TYPE_CHECKING:
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

    from sglang.srt.layers.radix_attention import RadixAttention

# The kernel contract of FlashInfer 0.7.4's cute-dsl-prims backend
# (flashinfer/attention/cute_dsl/sm120_fmha.py, SM120PrimsBatchPrefillBackend).
PRIMS_FP8_DTYPE = torch.float8_e4m3fn
PRIMS_HEAD_DIMS = (32, 64, 128, 256)
PRIMS_OUT_DTYPES = (torch.float16, torch.bfloat16)


def sm120_fp8_ragged_prefill_unsupported_reason(
    *,
    device_is_sm120: bool,
    kv_cache_dtype: torch.dtype,
    model_dtype: torch.dtype,
    head_dim: int,
    v_head_dim: int,
    num_qo_heads: int,
    num_kv_heads: int,
    has_wrapper_dispatch: bool,
    ragged_prefill_disabled: bool,
) -> Optional[str]:
    """Why the FP8 ragged prefill path cannot run for this model, or None if it can."""
    if not device_is_sm120:
        return "it requires an SM120 GPU (compute capability 12.0)"
    if kv_cache_dtype != PRIMS_FP8_DTYPE:
        return f"it requires --kv-cache-dtype fp8_e4m3, got {kv_cache_dtype}"
    if model_dtype not in PRIMS_OUT_DTYPES:
        return f"it requires a float16 or bfloat16 model, got {model_dtype}"
    if head_dim != v_head_dim or head_dim not in PRIMS_HEAD_DIMS:
        return (
            f"it requires equal QK/VO head dims in {PRIMS_HEAD_DIMS}, "
            f"got {head_dim}/{v_head_dim}"
        )
    if num_kv_heads <= 0 or num_qo_heads % num_kv_heads != 0:
        return (
            "it requires the query head count to be a multiple of the KV head "
            f"count, got {num_qo_heads}/{num_kv_heads}"
        )
    if has_wrapper_dispatch:
        return "sliding-window and encoder-decoder models are not supported"
    if ragged_prefill_disabled:
        return (
            "this configuration never uses the ragged prefill wrapper "
            "(paged-only, deterministic, multimodal or multi-item scoring)"
        )
    return None


@functools.lru_cache(maxsize=None)
def _scale_tensor(scale: float, device: torch.device) -> torch.Tensor:
    # One device copy per distinct per-layer scale, so the hot path never copies H2D.
    return torch.tensor([scale], dtype=torch.float32, device=device)


def quantize_to_fp8(x: torch.Tensor, scale: Optional[float]) -> torch.Tensor:
    # Same convention as the FP8 KV pool: store value / scale, saturated; one launch.
    scale_tensor = _scale_tensor(1.0 if scale is None else scale, x.device)
    return static_quant_fp8(x.contiguous(), scale_tensor)[0]


class SM120FP8RaggedPrefill:
    """One cute-dsl-prims ragged wrapper, planned alongside the regular ragged one."""

    def __init__(
        self,
        *,
        wrapper: BatchPrefillWithRaggedKVCacheWrapper,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        out_dtype: torch.dtype,
    ):
        self._wrapper = wrapper
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._out_dtype = out_dtype

    def plan(self, *, qo_indptr: torch.Tensor) -> None:
        # Causality and the softmax scale are baked into the plan. Planning with
        # sm_scale=1.0 lets each layer pass its own scaling through q_scale.
        self._wrapper.plan(
            qo_indptr,
            qo_indptr,
            self._num_qo_heads,
            self._num_kv_heads,
            self._head_dim,
            causal=True,
            sm_scale=1.0,
            q_data_type=PRIMS_FP8_DTYPE,
            kv_data_type=PRIMS_FP8_DTYPE,
            o_data_type=self._out_dtype,
        )

    def supports_layer(self, layer: RadixAttention) -> bool:
        return (
            layer.attn_type == AttentionType.DECODER
            and not layer.is_cross_attention
            and layer.sliding_window_size < 0
            and not layer.logit_cap
            and layer.tp_q_head_num == self._num_qo_heads
            and layer.tp_k_head_num == self._num_kv_heads
            and layer.head_dim == self._head_dim
            and layer.v_head_dim == self._head_dim
        )

    def forward(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        return_lse: bool,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # q: [num_tokens, num_qo_heads, head_dim]; k, v: [num_tokens, num_kv_heads,
        # head_dim]; all bf16/fp16. The LSE, when returned, is log2-based like FA2's.
        q_scale = layer.q_scale_float
        k_scale = layer.k_scale_float
        v_scale = layer.v_scale_float
        return self._wrapper.run(
            quantize_to_fp8(q, q_scale),
            quantize_to_fp8(k, k_scale),
            quantize_to_fp8(v, v_scale),
            q_scale=layer.scaling * (1.0 if q_scale is None else q_scale),
            k_scale=k_scale,
            v_scale=v_scale,
            return_lse=return_lse,
        )
