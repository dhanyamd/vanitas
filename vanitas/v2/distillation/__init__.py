"""Two-teacher distillation (S8 / Stage 4.6).

  - teacher_qwen3_32b.py — OpenRouter pipeline to collect Qwen3-32B logits/responses.
  - teacher_moshi.py     — (optional) rented-A100 pipeline to collect Moshi-7B STS traces.
  - distill_loss.py      — KL on text logits + token-NLL matching for audio traces.
"""
