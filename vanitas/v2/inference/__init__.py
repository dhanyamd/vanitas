"""Streaming inference + deployment.

  - streaming_server.py   — WebSocket front-end with KV-cache + depth-parallel decode.
  - speculative.py        — Qwen3-0.6B draft model for text-stream speculation.
  - constrained_decode.py — logit-processor glue (S5 in action at inference time).
  - gguf_export.py        — INT4 + GGUF artifact for llama.cpp / Mac deployment.
  - client_demo.py        — minimal mic-in / speaker-out client.
"""
