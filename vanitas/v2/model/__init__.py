"""Vanitas-SLM model components.

Modules (per PLAN.md \xa75):
  - backbone.py       — Qwen3-1.7B + 4 Mamba-2 layers (S2 surgery)
  - mamba_block.py    — the Mamba-2 block with zero-init residual gating
  - depth_decoder.py  — resolution-aware decoder for SNAC's 3 levels (S4)
  - cognition_gate.py — reasoning-mode routing gate (S6)
  - constraints.py    — logit processors for delay-pattern / valid-ID enforcement (S5)
  - personal_lora.py  — per-user LoRA adapters for on-device personalization (S7)
  - vanitas_slm.py    — multi-stream forward, ties everything together
"""
