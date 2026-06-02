# Vanitas — Locked Research Spec

> Decided 2026-06-02 after extensive literature survey (~20 papers) and scoping.
> This is the authoritative "what / why / what's ours" document.
> PLAN.md holds the engineering steps; this holds the research thesis.

## One-line thesis

**The first open, edge-deployable (<2B, on-device) speech model that reasons in
latent space *while listening* to the user, and continually learns its user at
inference through native memory — bringing two capabilities that today exist only
as recent papers (no usable code) into a working, reproducible, small system.**

## The honest framing

We are not claiming to *invent* latent reasoning or continual learning. We are
the first to **implement them, openly and reproducibly, in a small on-device
speech model** — a theory→practice contribution. Both target concepts are
**paper-only with no public implementation** (verified 2026-06-02):

- **Latent reasoning while listening** — FLAIR / "The Silent Thought"
  (arXiv:2603.17837, Apr 2026): paper-only, no code released.
- **Continual learning at inference for speech-to-speech** — theorized + done
  for ASR/text (test-time training), but **never realized for S2S**.

"No one has implemented it yet" is the opportunity, not a problem.

## NOT Moshi. NOT full-duplex.

- **Base is Qwen3-1.7B + SLAM-Omni recipe**, NOT Moshi (7B, cloud — violates the
  small/edge/cheap identity).
- We do **listening-time reasoning** (the model thinks during the user's turn),
  NOT Moshi-style talk-over-each-other full duplex (deferred / future work).
- The "duplex feeling" (no awkward pause, thinks while you speak) comes from
  streaming input + latent reasoning, achievable at edge scale.

## What we reuse (infrastructure — cited, frozen, not claimed)

| Component | Source | Role |
|---|---|---|
| Qwen3-1.7B | Alibaba, Apache-2 | the brain (intelligence) |
| SLAM-Omni recipe / SLAM-LLM | X-LANCE, MIT | S2S training scaffold (single-stage) |
| Whisper encoder | OpenAI | ears (audio → features) |
| SNAC / CosyVoice codec | MIT/Apache | mouth (tokens → waveform) |

## What we build ourselves (contributions — inspired, our own implementation)

| # | Contribution | Status | Risk |
|---|---|---|---|
| **C1 (headline)** | **Latent reasoning while listening** — recursive latent thinking during the user's speaking phase, FLAIR-inspired, our own implementation, at small/edge scale (no public code exists) | to build | 🔴 high (subtle method) |
| **C2** | **Continual learning at inference** — native bounded KV memory; relevance-gated writes, no backprop, no catastrophic forgetting; first for S2S | module built ✅, integrate | 🟡 medium |
| **Enabler** | **Mamba-hybrid Qwen3 backbone** — linear-time streaming; validated (PPL ratio 1.000) | done ✅ | 🟢 low |
| **C3 (eval)** | **Custom benchmarks**: H1 interaction quality (timing/interruption/recovery), H2 genuine vs post-hoc reasoning under streaming, H4 reasoning quality under incomplete input | to design | 🟡 medium |

## Scope discipline (the safety net — do not violate)

- **Must-ship:** working base (SLAM-Omni + Qwen3 + Mamba) + **C2 (continual memory)** + at least one custom benchmark (C3). C2 is more robust to build than C1, so it anchors the paper.
- **Headline stretch:** **C1 (latent reasoning while listening)** — the exciting, high-risk one. If it works → headline. If it fails → C2 + benchmarks still stand as a paper.
- **Order:** base → C2 (memory, robust) → C3 (benchmarks) → C1 (latent reasoning, risky) last so a failure doesn't sink everything.
- If C1 fails, the paper becomes "continual-learning S2S + benchmarks" — still novel, still publishable.

## Honest venue read (no false hope)

- Realistic A-tier: **Interspeech / ICASSP main, EMNLP/ACL Findings**.
- The **custom benchmarks (C3)** are the realistic **A+ vehicle** — NeurIPS/ICLR
  **Datasets & Benchmarks track** values exactly "the field lacks a benchmark for
  X; here's one + baselines." The field explicitly says these benchmarks are missing.
- A+ *main* track is a stretch (execution/scale bar is hard solo at $200) — possible
  only if C1 lands with a surprising result. Don't bank morale on it.
- No guarantees anywhere. The high floor: a working small system + a benchmark the
  field wants = a real contribution regardless of venue.

## Budget

~$120–180 (within $200). Base SFT ~$60–90; memory ~free; latent-reasoning training
modest; benchmarks are eval (cheap). Cheap RL/latent smoke tests (~$2) de-risk
before any big spend.

## The demo (abstract you can watch)

Laptop, offline. The user starts a question; the model is *visibly reasoning in
latent space while they speak* (a thinking indicator), answers with near-zero gap,
and across sessions recalls user facts — all on-device, sub-300 ms, no cloud.
