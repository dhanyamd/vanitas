# Vanitas — Locked Problem Statement

> Decided 2026-06-01 after extended scoping. This is the research thesis the
> project commits to. PLAN.md describes the engineering; this describes the
> *why* and the *what's novel*.

## One-sentence contribution

**A real-time speech language model that thinks asynchronously — generating its
response over *streaming* user input before the utterance completes, via an
RL-trained commit/wait/revise policy — and continually learns its user at
inference through external memory, with no backprop and no catastrophic
forgetting.**

Working title (pick at submission from results):
- *"Learning to Think Before You Speak: Reinforcement-Learned Asynchronous
  Generation for Streaming Speech Language Models"*
- *"Vanitas: RL-Trained Asynchronous Thinking with Continual Memory for
  Real-Time Speech"*

### Contribution structure (locked)

- **Headline (C1) — Asynchronous Thinking.** Streaming speech generation that
  starts responding before the user finishes, then revises as more audio
  arrives. A new *capability* for speech LMs, directly cashing out the latency
  goal (can reach ~zero / negative effective latency).
- **Method for C1 — RL.** The core of async thinking is a sequential decision
  under partial information: *commit / wait / revise*. Trained with RL (GRPO /
  RLVR) on a verifiable reward = response quality − latency − revision cost.
  This is *why RL is in the title* — it is the training method for the headline
  capability, not a bolted-on extra.
- **Contribution 2 (C2) — Continual learning at inference.** Memory-based,
  non-parametric: the model writes experiences to an external memory and
  retrieves at inference. No backprop, no catastrophic forgetting (weights
  frozen). Under-explored for speech.
- **Substrate — Compute-Memory Separation.** Small frozen Qwen3 (compute) +
  growing external memory (knowledge). Enables C2.
- **Motivation only (not a deliverable) — Evaluation ∝ GDP / dynamic test-time
  eval.** North-star framing; too broad to operationalize in this paper.

Maps to the project's four-pillar vision: async thinking = pillar 2, continual
learning = pillar 4, compute-memory separation = pillar 1, GDP-eval = pillar 3
(motivation only).

### Scope discipline

- **Must-ship:** the RL-trained asynchronous-thinking streaming model on the
  working SLAM-Omni + Qwen3 + Mamba base. This alone is the paper.
- **C2 (memory continual learning):** cheap; add once C1 works.
- Base build is identical regardless of headline → proceeds now, zero waste.

## Why it's novel (pressure-tested against the literature)

- Continual learning in speech exists **only for ASR and TTS speaker-adaptation**
  (e.g. arXiv:2506.16574, arXiv:2103.14512). Nobody has studied continual
  personalization of a full **speech-to-speech assistant**.
- "First to study X" is the strongest claim a solo / low-budget researcher can
  make. Problem-formulation + benchmark + baseline papers are disproportionately
  cited and accepted because they open a direction rather than competing on
  compute.

## The reinforcing loop (why the pieces need each other)

```
        On-device (edge)  ──makes──►  personalization SAFE (privacy)
              ▲                                  │
              │                                  ▼
   makes edge WORTH IT          personalization needs LOCAL data
              │                                  │
              └──────────  Vanitas  ◄────────────┘
                   (latency / quality / intelligence
                    are the constraints that make it usable)
```

- **Edge** makes personalization *safe* — you'd never let a cloud assistant deeply
  learn your private life, but a fully-local one you would.
- **Personalization** makes edge *worth it* — why run locally if the model is generic?
- **Latency, quality, intelligence** are properties the system must *maintain*, not
  separate contributions.

## Two contributions

1. **Continual on-device personalization for speech-to-speech** — the novel
   problem (unstudied for S2S). *Headline.*
2. **RL (GRPO/RLVR) for the latency–quality tradeoff** — keeps a personalizing
   assistant fast + intelligible. Verifiable reward = ASR-intelligibility +
   brevity. Trendy (post-R1 RLVR wave), and has a safe floor (always yields a
   publishable Pareto curve).

The **Mamba-hybrid backbone** is the engineering enabler (linear-time streaming →
real-time on edge), not the headline — Jamba/Zamba already established
Mamba-Transformer hybrids. **Edge + privacy** is the motivation + demo.

Why RL is load-bearing, not bolted on: when the model personalizes (learns user
facts), responses get longer/slower. RL with an intelligibility+brevity reward is
the mechanism that keeps a personalizing assistant real-time. RL *serves* P1.

## Scope discipline (committed vs stretch)

| Phase | What | Commitment |
|---|---|---|
| **Must-ship (the paper)** | Working S2S base (SLAM-Omni recipe + Qwen3-1.7B + Mamba) + **P1 continual personalization + benchmark** + edge (GGUF) deployment | committed |
| **Contribution 2 (primary stretch)** | **RL (GRPO/RLVR)** for latency–quality; pursue right after P1 works | strong intent |
| Stretch (last) | **P2**: latent "think-before-speak" reasoning | only if P1 + RL land |

If only P1 ships, it is still a novel, publishable, A+-shot paper. RL and P2 are
upgrades, never dependencies.

## The demo (the "abstract you can watch")

Laptop in **airplane mode**. Three conversation sessions. By session 3 the
assistant greets the user by name, recalls a preference stated in session 1,
and has subtly matched the user's speaking pace — all offline, sub-300 ms,
no network. The 90-second clip proves edge + privacy + personalization +
latency + quality simultaneously.

## Properties to maintain (hard constraints, measured in eval)

- **Latency:** TTFA ≤ 300 ms on consumer hardware (target ≤ 200 ms on a 4090).
- **Intelligence:** inherited from Qwen3-1.7B; must not regress after personalization.
- **Speech quality:** UTMOS ≥ 3.2; generated speech intelligible (Whisper-WER ≤ 25%).
- **No catastrophic forgetting:** general-task quality drops ≤ 2 pts after personalization.
- **Edge:** INT4 GGUF, ≤ 8 GB, runs in llama.cpp on a Mac.

## Venue targets (honest)

- Realistic A-tier: **Interspeech / ICASSP main**, **EMNLP/ACL Findings**.
- Outside shot at A+: **EMNLP/ACL/NeurIPS main** if the forgetting analysis or
  the personalization method is sharp/surprising.
- No guarantees at any venue — acceptance is novelty × execution × reviewer luck.
  P1's value is the high floor: a benchmark + framing is publishable even if the
  method is simple.

## Budget

~$150–180 (under the $200 ceiling). SFT base + personalization experiments are
cheap (LoRA, simulated users). P2/P3 add cost only if pursued.

## A+ reach angles — parked, decided LATER with evidence

Build the planned system first (P1 → P2 → P3). Once the base + personalization
data exist, evaluate whether to chase one of these for a top-tier (NeurIPS/ACL
main) push. We decide with measured evidence, not now.

- ⭐ **Candidate A (NEW LEAD) — Predict-and-revise: responding before the user
  finishes.** The model begins generating its spoken response from *partial*
  user input and continuously revises as more audio streams in — so when the
  user stops, the reply is already formed (sometimes *negative* latency). A new
  *capability* (proactive, revisable speech generation), not engineered novelty.
  Reframes latency from "respond fast after they finish" to "start before they
  finish and be right." Honest stepping-stone to full-duplex without the full
  cost. Hard ML problem (generate under uncertainty about utterance end +
  revise) → genuine A+ depth. Unexplored for speech LMs.
- **Candidate A2 (scientific, safer fallback) — does reasoning survive the
  text→speech modality shift?** Measure whether a text LLM's reasoning degrades
  when adapted to speak, why, and how to preserve it. Cheap (measurement), lower
  risk, less flashy but A+ venues like clean scientific findings.
- **Candidate A3 (retired as too toy) — RL-gated latent reasoning ("when to
  think").** The "hi vs math problem" framing is engineered novelty, not
  attention-worthy. Kept only as a possible *component* of predict-and-revise.
- **Candidate B — RL-recovered quantization.** Aggressive INT4 flattens prosody
  before content; use RL to recover expressive quality on the quantized edge
  model. Ties size + RL. Studies quantization's differential effect on
  acoustic vs semantic tokens.
- **Candidate C — personalization without weight updates.** Memory-augmented
  speech LM that personalizes by reading an external personal memory → zero
  catastrophic forgetting by design (architectural, not a regularizer).

Rejected as too generic: "what does a speech LM forget during personalization."
