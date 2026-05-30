# Vanitas-SLM (v2)

The from-scratch (mostly) rebuild. See `../../PLAN.md` for the locked design,
training curriculum, contributions, budget, and timeline.

## Status

Scaffolded. Code lives under `vanitas.v2.*`. V1 code under `vanitas/{model,training,...}/`
is still present, untouched, and runnable. Once v2 Stage 0 passes, v1 can be archived.

## Layout

```
vanitas/v2/
  codec/        — SNAC wrapper (frozen, MIT)
  tokenizer/    — Qwen3 vocab expansion utilities
  model/        — backbone surgery, depth decoder, gates, constraints, personal LoRA
  data/         — synth duplex pipeline, SNAC encode, stream interleaving
  training/     — one script per stage (1, 2, 3, 4.5, 4.6, 4.7, 5, 5.5)
  distillation/ — S8 two-teacher distillation pipeline
  continual/    — S7 online LoRA + EWC + replay
  eval/         — latency, FD-Bench, MTR-Bench, UTMOS, personalization, Moshi comparison
  inference/    — streaming server, GGUF export, speculative decode
```

## Sanity check (run first)

From the repo root:

```bash
# Mac (SNAC only — Qwen3 LM check needs CUDA):
python scripts/v2/stage0_sanity.py --skip-llm

# Any CUDA box:
python scripts/v2/stage0_sanity.py
```

Exit code 0 → ready for Stage 1. Anything else → fix before proceeding.

## Dependencies you will need

Not yet wired into `pyproject.toml`. To install ad-hoc into your existing venv:

```bash
pip install snac transformers
# For training (CUDA only):
pip install unsloth mamba-ssm bitsandbytes peft accelerate
```

The Stage 0 sanity script only needs `snac` and `transformers`.

## What is NOT here yet

This is scaffold-only. Stage 1+ code is empty stubs in `__init__.py` docstrings.
That's intentional — write each stage when you reach it, not before.

When implementing a stage, follow PLAN.md §3 for the spec and §6 for the exit criterion.
