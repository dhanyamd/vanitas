"""SNAC codec wrapper.

SNAC (Hubert Siuzdak, MIT) is a hierarchical RVQ neural audio codec at 24 kHz.
It emits 3 resolution levels at ~12, 23, and 47 Hz. We use it frozen.

Reference: https://github.com/hubertsiuzdak/snac
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"
SNAC_SAMPLE_RATE = 24_000


@dataclass(frozen=True)
class SnacOutput:
    """One SNAC encode result.

    ``codes`` is a list of 3 LongTensors with shapes:
      level0: (B, T_12hz)
      level1: (B, T_23hz)   # ~2x level0
      level2: (B, T_47hz)   # ~4x level0
    """
    codes: list[torch.Tensor]

    @property
    def num_levels(self) -> int:
        return len(self.codes)

    def to(self, device) -> "SnacOutput":
        return SnacOutput([c.to(device) for c in self.codes])


def load_snac(device: str = "auto"):
    """Load the pretrained SNAC model in eval mode.

    Returns the SNAC instance. ``device="auto"`` picks cuda → mps → cpu.
    """
    try:
        from snac import SNAC
    except ImportError as exc:
        raise ImportError(
            "snac is not installed. Run `pip install snac` (or add it to pyproject.toml)."
        ) from exc

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = SNAC.from_pretrained(SNAC_MODEL_ID).eval().to(device)
    # SNAC's parameters are frozen for our use; turn off grad to be explicit.
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def encode(model, waveform: torch.Tensor) -> SnacOutput:
    """Encode 24 kHz mono audio.

    Args:
        model: SNAC instance from ``load_snac``.
        waveform: (B, 1, T) float tensor at 24 kHz, range [-1, 1].
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(1)
    waveform = waveform.to(next(model.parameters()).device)
    codes: Sequence[torch.Tensor] = model.encode(waveform)
    return SnacOutput(codes=[c for c in codes])


@torch.no_grad()
def decode(model, snac_out: SnacOutput) -> torch.Tensor:
    """Decode SNAC codes back to a 24 kHz waveform of shape (B, 1, T)."""
    device = next(model.parameters()).device
    codes = [c.to(device) for c in snac_out.codes]
    return model.decode(codes)
