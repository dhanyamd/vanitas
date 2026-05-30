"""Vanitas-SLM v2.

The from-scratch (mostly) rebuild on top of:
  - Qwen3-1.7B (Apache-2) as backbone, with 4 Mamba-2 layers surgically inserted.
  - SNAC (MIT) as a frozen audio codec.
  - Unsloth (QLoRA) as the training framework.

Layout mirrors PLAN.md \xa75 with two differences for the conservative scaffold:
  - This whole package sits under ``vanitas.v2`` so v1 keeps working until removed.
  - ``audio/`` from v1 is reused via re-export; not duplicated.

When v1 is archived, the contents of this package move up one level to ``vanitas/``.
"""
