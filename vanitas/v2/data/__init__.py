"""Data pipeline.

Modules (per PLAN.md \xa75):
  - synth_duplex.py — Stage 3 corpus builder (Qwen + Kokoro + stereo mix)
  - snac_encode.py  — offline SNAC encoding of audio
  - streams.py      — delay-pattern interleaving for the 3-stream LM format (S3)
"""
