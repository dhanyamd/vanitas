"""Small zero-initialized adapters for customer/domain fine-tuning."""
from __future__ import annotations

import torch
import torch.nn as nn


class LowRankAdapter(nn.Module):
    """LoRA-style residual adapter for sequence states.

    The up projection is zero-initialized, so adding the adapter does not change
    the base model at initialization. Later fine-tuning can train only adapter
    parameters for domain-specific conversational behavior.
    """

    def __init__(self, dim: int, rank: int = 8, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.up(self.down(x))
