"""On-device continual personalization (S7 / Stage 4.5 + 5.5).

  - online_lora.py     — per-user LoRA hot-swap + online update loop.
  - ewc.py             — Elastic Weight Consolidation against general-LoRA Fisher.
  - replay_buffer.py   — small replay buffer of general-domain samples.
  - user_simulator.py  — 50 synthetic users with distinct vocab + voice patterns.
"""
