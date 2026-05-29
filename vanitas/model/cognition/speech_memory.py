"""Speech-native memory encoder.

This module creates trainable memory targets directly from speech features. It
lets Vanitas learn an internal speech-memory space instead of relying only on
external text encoders.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeechMemoryEncoder(nn.Module):
    """Encodes mel sequences into a single normalized memory vector."""

    def __init__(self, mel_dim: int = 80, memory_dim: int = 512, hidden_dim: int = 512, prosody_dim: int = 8):
        super().__init__()
        self.frame_proj = nn.Sequential(
            nn.Linear(mel_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, memory_dim),
        )
        self.prosody_proj = nn.Sequential(
            nn.Linear(prosody_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        mel: torch.Tensor,
        lengths: torch.Tensor | None = None,
        prosody_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one normalized memory embedding per sequence.

        Args:
            mel: Tensor of shape (B, T, n_mels).
            lengths: Optional non-padded sequence lengths, shape (B,).
        """
        B, T, _ = mel.shape
        h = self.frame_proj(mel)
        logits = self.attn_pool(h).squeeze(-1)

        if lengths is not None:
            positions = torch.arange(T, device=mel.device).unsqueeze(0)
            valid = positions < lengths.to(mel.device).unsqueeze(1)
            logits = logits.masked_fill(~valid, -1e4)

        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        pooled = (h * weights).sum(dim=1)
        if prosody_features is not None:
            pooled = pooled + self.prosody_proj(prosody_features.to(mel.device, dtype=mel.dtype))
        embedding = self.out(pooled)
        return F.normalize(embedding, p=2, dim=-1)
