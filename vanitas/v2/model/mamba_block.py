"""Mamba-2 residual block for surgery into Qwen3 (S2 in PLAN.md).

Design choices and why:

  * **Zero-init residual gate.** A single learnable scalar ``gate`` is
    initialized to 0. At training start, the block contributes nothing to
    the residual stream, so the modified Qwen3 is bitwise-equivalent to base
    Qwen3 at step 0. Distillation in Stage 1 gradually opens the gate.
    This is the cleanest way to do the surgery without destroying base
    quality. (Alternative: zero-init the Mamba out_proj. We don't do that
    because it would void the internal SSD parametrization initialization
    upstream cares about.)

  * **RMSNorm before Mamba.** Qwen3's blocks use RMSNorm, so we match it.

  * **HF transformer-block interface.** ``forward(hidden_states, **kwargs)``
    returns a 1-tuple of ``(hidden_states,)``. This matches the shape that
    Qwen3DecoderLayer.forward returns when ``output_attentions=False`` and
    ``use_cache=False`` — which is the only mode we use during Stage 1
    distillation. Cache support for streaming inference is added in Stage 6.

  * **CUDA only at runtime.** ``mamba-ssm`` ships CUDA kernels only. On Mac
    or CPU, importing this file works but instantiating the block raises a
    clear error. Stage 1 must run on CUDA.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


_MAMBA_IMPORT_ERROR: Optional[BaseException] = None
try:
    from mamba_ssm.modules.mamba2 import Mamba2 as _OfficialMamba2  # type: ignore
except Exception as exc:  # pragma: no cover  -- environmental
    _OfficialMamba2 = None  # type: ignore[assignment]
    _MAMBA_IMPORT_ERROR = exc


def mamba_available() -> bool:
    """True iff the official CUDA Mamba-2 kernels can be imported here."""
    return _OfficialMamba2 is not None


class RMSNorm(nn.Module):
    """RMSNorm matching Qwen3's variant (eps=1e-6, no centering)."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for stability, cast back to input dtype.
        in_dtype = x.dtype
        x32 = x.float()
        variance = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(variance + self.eps)
        return (x32.to(in_dtype)) * self.weight


class Mamba2ResidualBlock(nn.Module):
    """Mamba-2 SSD block, residual-gated, drop-in for one Qwen3 decoder layer.

    Shape contract:
        Input  hidden_states: (B, L, d_model)
        Output hidden_states: (B, L, d_model)

    At init, output ≡ input (gate=0). After Stage 1 distillation, gate is
    learned per layer.

    Args:
        d_model:   model hidden size (must match Qwen3's hidden_size).
        d_state:   SSM state size. 128 matches Mamba-2's default.
        d_conv:    1-D causal conv kernel size. 4 = Mamba-2 default.
        expand:    inner expansion factor. 2 = Mamba-2 default.
        layer_idx: which Qwen3 layer slot we are replacing — for debug only.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        if not mamba_available():
            raise RuntimeError(
                "mamba-ssm is not installed (CUDA kernels required). "
                f"Original import error: {_MAMBA_IMPORT_ERROR!r}\n"
                "This block is importable for inspection on Mac/CPU, but "
                "instantiating it requires a CUDA environment with "
                "`pip install mamba-ssm`."
            )

        self.d_model = d_model
        self.layer_idx = layer_idx
        # Qwen3Model.forward looks up causal_mask_mapping[layer.attention_type]
        # BEFORE calling each layer. We need to carry an attribute it accepts
        # ("full_attention" or "sliding_attention"). We pick "full_attention" so
        # Qwen3 builds a standard causal mask, then ignore it inside our forward.
        self.attention_type = "full_attention"
        self.norm = RMSNorm(d_model)
        # The official Mamba-2 block already projects back to d_model internally.
        self.mamba = _OfficialMamba2(  # type: ignore[misc]
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        # Scalar residual gate. Initialized to 0 ⇒ surgery is identity at step 0.
        # Stored as a scalar nn.Parameter so it shows up in optimizer + checkpoints.
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Any = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        """Run a Mamba-2 residual step.

        Most kwargs are accepted for HF-decoder-layer compatibility but
        ignored — SSMs don't use attention masks or position ids.
        """
        residual = hidden_states
        x = self.norm(hidden_states)
        x = self.mamba(x)
        out = residual + self.gate * x

        # HF Qwen3DecoderLayer.forward returns:
        #   (hidden_states, [self_attn_weights], [present_key_value])
        # We only ever return the first element since we never produce
        # attentions or KV cache from a Mamba block.
        outputs: tuple[torch.Tensor, ...] = (out,)
        if output_attentions:
            outputs = outputs + (None,)  # type: ignore[arg-type]
        if use_cache:
            outputs = outputs + (None,)  # type: ignore[arg-type]
        return outputs


# ---------------------------------------------------------------------------
# Sanity helpers
# ---------------------------------------------------------------------------

def parameter_count(d_model: int, d_state: int = 128, expand: int = 2) -> int:
    """Approximate Mamba-2 block parameter count for budget planning.

    Not exact (depends on Mamba-2 internal kernels), but close enough to
    confirm our 'add 4 × ~30M = ~120M params' claim in PLAN.md.
    """
    d_inner = expand * d_model
    nheads = d_inner // 64  # standard Mamba-2 head dim
    # Input projection: d_model -> d_inner + d_state + d_state + nheads + d_inner
    in_proj = d_model * (d_inner + 2 * d_state + nheads + d_inner)
    # Output projection: d_inner -> d_model
    out_proj = d_inner * d_model
    # Conv1d (depthwise): (d_inner + 2*d_state) * 4
    conv = (d_inner + 2 * d_state) * 4
    # RMSNorm + dt_bias + A_log + scalar gate
    other = d_model + nheads + nheads + 1
    return in_proj + out_proj + conv + other


if __name__ == "__main__":
    # Importable-but-not-instantiable sanity check.
    print(f"mamba-ssm available: {mamba_available()}")
    for d in (1024, 1536, 2048, 2560):
        n = parameter_count(d_model=d)
        print(f"  d_model={d:>4}  approx Mamba-2 block params ≈ {n:>11,d}  (~{n/1e6:.1f}M)")
