"""Shared training utilities used across all Stage 1–5.5 scripts.

Three responsibilities:

  1. **Model loading.** Tries Unsloth's fast 4-bit QLoRA loader first; falls
     back to vanilla HuggingFace transformers if Unsloth is unavailable
     (Mac, CPU envs, or stages where we explicitly want fp16 not 4-bit).
  2. **Distillation loss.** KL(student || teacher) over the vocab, computed
     with temperature, in fp32 for stability.
  3. **Logging / checkpoint helpers.** Light wrappers, not a framework.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def unsloth_available() -> bool:
    try:
        import unsloth  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def load_qwen3_for_training(
    model_id: str = "Qwen/Qwen3-1.7B",
    use_qlora: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    max_seq_length: int = 4096,
):
    """Return (model, tokenizer) ready for training.

    If Unsloth + ``use_qlora`` are available we get a 4-bit student in ~6 GB.
    Otherwise we fall back to fp16 HF transformers (~3.5 GB at fp16, but no
    gradient memory savings; only suitable for distillation on big GPUs).
    """
    if use_qlora and unsloth_available():
        from unsloth import FastLanguageModel  # type: ignore

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            dtype=None,  # let Unsloth pick the best dtype for the device
            load_in_4bit=True,
        )
        return model, tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Distillation loss
# ---------------------------------------------------------------------------

def kl_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL(student || teacher) over the vocab.

    Standard knowledge distillation. Computed in fp32 for stability.
    Returns a scalar mean-reduced over the masked positions.

    Args:
        student_logits: (B, L, V) — student's per-token logits.
        teacher_logits: (B, L, V) — teacher's per-token logits.
        temperature:    softens both distributions; higher = more uniform.
        mask:           (B, L) bool/0-1; True/1 positions are included.
    """
    s = student_logits.float() / temperature
    t = teacher_logits.float() / temperature

    log_p_s = F.log_softmax(s, dim=-1)
    p_t = F.softmax(t, dim=-1)
    # token-level KL: -sum_v p_t * log_p_s  (constant entropy term ignored).
    per_tok = -(p_t * log_p_s).sum(dim=-1)  # (B, L)

    if mask is None:
        loss = per_tok.mean()
    else:
        m = mask.float()
        loss = (per_tok * m).sum() / m.sum().clamp(min=1.0)

    # The standard KD scaling: multiply by T^2 so loss magnitude does not vanish
    # at high temperature.
    return loss * (temperature ** 2)


# ---------------------------------------------------------------------------
# LR schedules
# ---------------------------------------------------------------------------

def cosine_warmup_schedule(
    step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.0
) -> float:
    """Standard linear warmup followed by half-cosine decay. Returns multiplier in [0, 1]."""
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cos


# ---------------------------------------------------------------------------
# Checkpoint + manifest
# ---------------------------------------------------------------------------

@dataclass
class TrainingManifest:
    """Persistent record of a stage run, written next to the checkpoint."""
    stage: str
    step: int
    total_steps: int
    loss: float
    val_metric: float | None
    wallclock_s: float
    config: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def save_training_checkpoint(
    out_dir: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    manifest: TrainingManifest,
    tokenizer=None,
) -> None:
    """Save trainable parameters + optimizer state + manifest to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save only trainable parameters — base Qwen3 weights are not ours to redistribute
    # at every checkpoint, and saving the full 1.7B on every step is wasteful.
    trainable_sd = {
        k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad
    }
    torch.save(trainable_sd, out / "trainable_params.pt")

    if optimizer is not None:
        torch.save(optimizer.state_dict(), out / "optimizer.pt")

    if tokenizer is not None:
        tokenizer.save_pretrained(out)

    (out / "manifest.json").write_text(manifest.to_json())


# ---------------------------------------------------------------------------
# Tiny logging wrapper
# ---------------------------------------------------------------------------

class StepLogger:
    """Prints progress every ``log_every`` steps with running averages."""

    def __init__(self, log_every: int = 25, stage: str = "?") -> None:
        self.log_every = log_every
        self.stage = stage
        self.t0 = time.time()
        self._losses: list[float] = []

    def log(self, step: int, loss: float, lr: float, extra: dict[str, Any] | None = None) -> None:
        self._losses.append(loss)
        if step % self.log_every == 0:
            avg = sum(self._losses[-self.log_every:]) / min(self.log_every, len(self._losses))
            elapsed = time.time() - self.t0
            extra_str = " | ".join(f"{k}={v}" for k, v in (extra or {}).items())
            print(
                f"[{self.stage} step {step}] loss={loss:.4f}  avg{self.log_every}={avg:.4f}  "
                f"lr={lr:.2e}  elapsed={elapsed:.0f}s"
                + (f"  {extra_str}" if extra_str else "")
            )
