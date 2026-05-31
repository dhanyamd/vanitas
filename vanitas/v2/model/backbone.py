"""Vanitas-SLM hybrid backbone — Qwen3 with Mamba-2 layers inserted (S2).

This module is the construction site of the surgery from PLAN.md §2.2.

  Qwen3-1.7B has 28 decoder layers. We replace 4 of them:

      L0 L1 L2 L3 L4 L5 [M] L7 L8 L9 L10 L11 L12 [M]
      L14 L15 L16 L17 L18 L19 [M] L21 L22 L23 L24 L25 L26 [M]

  where [M] is a :class:`Mamba2ResidualBlock` with a zero-init residual
  gate so the surgical student is bitwise-equal to base Qwen3 at step 0.

Three functions form the public API:

  * :func:`load_base_qwen3` — fetch Qwen3-1.7B as a frozen teacher.
  * :func:`build_vanitas_backbone` — load Qwen3-1.7B and apply surgery in
    one shot, returning the student model ready for Stage 1 distillation.
  * :func:`resize_for_expanded_vocab` — grow embed + lm_head for the new
    SNAC vocab without disturbing the pretrained rows.

Heavy operations (loading the 1.7B weights, mamba surgery) require ~6 GB
of RAM in fp16; surgery on CUDA also needs ``mamba-ssm`` installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from vanitas.v2.model.mamba_block import Mamba2ResidualBlock, mamba_available


QWEN3_MODEL_ID = "Qwen/Qwen3-1.7B"
DEFAULT_MAMBA_POSITIONS: tuple[int, ...] = (6, 13, 20, 27)


@dataclass
class SurgeryReport:
    """What we did to the base model, for logging + checkpoint metadata."""
    base_model_id: str
    num_layers: int
    mamba_positions: tuple[int, ...]
    mamba_d_state: int
    mamba_d_conv: int
    mamba_expand: int
    base_vocab_size: int
    new_vocab_size: int | None
    total_params: int
    trainable_params: int

    def summary(self) -> str:
        return (
            f"Vanitas-SLM backbone\n"
            f"  base:           {self.base_model_id}\n"
            f"  layers:         {self.num_layers} total, {len(self.mamba_positions)} replaced with Mamba\n"
            f"  mamba positions: {self.mamba_positions}\n"
            f"  mamba config:   d_state={self.mamba_d_state}, d_conv={self.mamba_d_conv}, expand={self.mamba_expand}\n"
            f"  vocab:          {self.base_vocab_size:,}"
            + (f" → {self.new_vocab_size:,}" if self.new_vocab_size else "")
            + f"\n  total params:   {self.total_params:,} ({self.total_params/1e9:.2f}B)\n"
            f"  trainable:      {self.trainable_params:,} ({self.trainable_params/1e6:.1f}M)"
        )


# ---------------------------------------------------------------------------
# Teacher loading (frozen, for distillation)
# ---------------------------------------------------------------------------

def load_base_qwen3(
    model_id: str = QWEN3_MODEL_ID,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
) -> nn.Module:
    """Load Qwen3-1.7B unmodified, frozen, eval-mode. Used as the Stage-1 teacher.

    Args:
        device: target device. If ``None``, picks cuda → mps → cpu.
                Passing a string ("cuda", "cpu") or a ``torch.device`` also works.
    """
    from transformers import AutoModelForCausalLM

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # HF transformers ≥ 4.45 renames the kwarg `torch_dtype` → `dtype`;
    # we try the new name first and fall back for older installs.
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Student: Qwen3 + Mamba surgery
# ---------------------------------------------------------------------------

def apply_mamba_surgery(
    model: nn.Module,
    positions: Sequence[int] = DEFAULT_MAMBA_POSITIONS,
    d_state: int = 128,
    d_conv: int = 4,
    expand: int = 2,
) -> None:
    """Insert Mamba-2 residual blocks *after* the listed Qwen3 decoder layers.

    Note this is INSERT, not REPLACE. The original Qwen3 layers are all
    preserved; we add 4 new Mamba blocks in between them. With the Mamba
    block's gate initialized to zero, each new block is the identity
    function at step 0, so the student is bitwise-equivalent to the
    unmodified base model. Distillation then opens the gates to let the
    Mamba blocks contribute.

    Earlier versions of this function replaced layers, which dropped 4 of
    Qwen3's 28 attention/FFN blocks and roughly doubled PPL at init —
    distillation could in principle recover, but it's much cleaner to
    start at teacher quality and learn upward from there.

    Example with positions=(6, 13, 20, 27) on the 28-layer Qwen3-1.7B
    stack produces a 32-layer model with Mamba blocks at indices
    7, 15, 23, 31 (immediately after each requested position).

    Mutates ``model`` in place; does not return anything.
    """
    if not positions:
        return

    if not mamba_available():
        raise RuntimeError(
            "Cannot apply Mamba surgery without CUDA mamba-ssm installed."
        )

    # Qwen3's HF layout: model.model.layers is a ModuleList of decoder layers.
    layers: nn.ModuleList = model.model.layers  # type: ignore[attr-defined]
    n_layers = len(layers)
    d_model = model.config.hidden_size

    bad = [p for p in positions if not 0 <= p < n_layers]
    if bad:
        raise ValueError(
            f"Insert-after positions out of range for {n_layers}-layer model: {bad}"
        )

    if len(set(positions)) != len(positions):
        raise ValueError(f"Duplicate Mamba positions are not allowed: {positions}")

    # Match the original layers' dtype / device on the new blocks.
    template_param = next(layers[positions[0]].parameters())
    dtype = template_param.dtype
    device = template_param.device

    insert_after = set(positions)

    new_layers: list[nn.Module] = []
    for i, layer in enumerate(layers):
        new_layers.append(layer)
        if i in insert_after:
            mamba_block = Mamba2ResidualBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                layer_idx=len(new_layers),  # position in the new list
            ).to(device=device, dtype=dtype)
            new_layers.append(mamba_block)

    model.model.layers = nn.ModuleList(new_layers)  # type: ignore[attr-defined]
    # Update the model config so HF's loops and KV-cache logic see the right depth.
    model.config.num_hidden_layers = len(new_layers)


def resize_for_expanded_vocab(
    model: nn.Module,
    new_vocab_size: int,
    init_std: float = 0.02,
) -> None:
    """Grow embed + lm_head to ``new_vocab_size``.

    HF's ``resize_token_embeddings`` handles both ends and reuses pretrained
    rows for old token ids; we re-init the new rows to N(0, init_std**2) since
    HF's default copy-mean-of-existing tends to bias new audio tokens toward
    English semantics, which we explicitly do not want.
    """
    old_size = model.get_input_embeddings().weight.shape[0]
    if new_vocab_size == old_size:
        return
    if new_vocab_size < old_size:
        raise ValueError("Refusing to shrink vocab. That would drop pretrained tokens.")

    model.resize_token_embeddings(new_vocab_size)

    with torch.no_grad():
        embed = model.get_input_embeddings().weight
        embed[old_size:].normal_(mean=0.0, std=init_std)

        out = model.get_output_embeddings()  # lm_head
        if out is not None and not getattr(model.config, "tie_word_embeddings", False):
            out.weight[old_size:].normal_(mean=0.0, std=init_std)


def build_vanitas_backbone(
    model_id: str = QWEN3_MODEL_ID,
    mamba_positions: Sequence[int] = DEFAULT_MAMBA_POSITIONS,
    mamba_d_state: int = 128,
    mamba_d_conv: int = 4,
    mamba_expand: int = 2,
    new_vocab_size: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
) -> tuple[nn.Module, SurgeryReport]:
    """Load Qwen3-1.7B, apply surgery, optionally expand vocab. One-shot helper."""
    from transformers import AutoConfig, AutoModelForCausalLM

    base_cfg = AutoConfig.from_pretrained(model_id)
    n_layers = base_cfg.num_hidden_layers
    base_vocab_size = base_cfg.vocab_size

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.train()

    apply_mamba_surgery(
        model,
        positions=mamba_positions,
        d_state=mamba_d_state,
        d_conv=mamba_d_conv,
        expand=mamba_expand,
    )

    if new_vocab_size is not None:
        resize_for_expanded_vocab(model, new_vocab_size)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    report = SurgeryReport(
        base_model_id=model_id,
        num_layers=n_layers,
        mamba_positions=tuple(mamba_positions),
        mamba_d_state=mamba_d_state,
        mamba_d_conv=mamba_d_conv,
        mamba_expand=mamba_expand,
        base_vocab_size=base_vocab_size,
        new_vocab_size=new_vocab_size,
        total_params=total,
        trainable_params=trainable,
    )
    # Print once so logs always include the model summary; callers don't have
    # to remember to do this.
    print(report.summary())
    return model, report


# ---------------------------------------------------------------------------
# Dry-run on CPU/MPS (no Mamba)
# ---------------------------------------------------------------------------

def dryrun_layer_map(model_id: str = QWEN3_MODEL_ID) -> None:
    """Print the layer map without instantiating Mamba.

    Reflects INSERT semantics: Mamba blocks are inserted immediately after
    each of ``DEFAULT_MAMBA_POSITIONS``, growing the stack by 4 layers.

    Useful on Mac to confirm positions / counts before sending a CUDA job.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    n_layers = cfg.num_hidden_layers
    insert_after = set(DEFAULT_MAMBA_POSITIONS)

    new_total = n_layers + len(insert_after)
    print(
        f"Model: {model_id}  ({n_layers} base layers → {new_total} after "
        f"INSERT, hidden={cfg.hidden_size})"
    )
    new_idx = 0
    for i in range(n_layers):
        print(f"  new L{new_idx:>2}  ← Qwen3 L{i}")
        new_idx += 1
        if i in insert_after:
            print(f"  new L{new_idx:>2}  [Mamba-2 inserted after Qwen3 L{i}]")
            new_idx += 1


if __name__ == "__main__":
    dryrun_layer_map()
