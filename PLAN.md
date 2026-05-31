# Vanitas-SLM — Final Plan (Locked, Ambitious Edition)

> **Title:** Vanitas-SLM: An Edge-Deployable 1.8B Full-Duplex Speech Language Model with Adaptive Reasoning, On-Device Personalization, and Distillation-Closed Quality Gap
> **Status:** Locked design with two ambitious bets (B + E). May 2026.
> **Authors:** dhanyamd + collaborator
> **Budget:** $380 floor (with Kaggle/Colab/Modal credits stacked) — $725 ceiling
> **Timeline:** 9 weeks
> **Venue target:** Interspeech 2026 → ASRU 2026 → EMNLP 2026 → NeurIPS 2026 (in that order)

---

## 0. Scope decisions (locked)

- **Small** — total inference params ~1.8B. INT4 GGUF deploys in ~1 GB.
- **Pretrained components accepted as infrastructure** — Qwen3-1.7B (Apache-2) and SNAC (MIT). Cited, not claimed.
- **Full duplex non-negotiable** — true parallel user+agent streams.
- **Sub-200 ms TTFA non-negotiable** — target 180 ms on RTX 4090, 220 ms on Mac mini M3.
- **Eight contributions** ranked by risk. S1–S6 are the safe paper. S7 (continual learning) and S8 (Moshi-gap distillation) are the ambitious bets that elevate the paper if they land. Plan survives if either or both fail.
- **Decision gate at end of week 6** decides which big bet(s) get full effort.

### Out of scope (intentionally — do not re-add)

These were considered and explicitly rejected. Reasons logged here so a future
self doesn't litigate them again mid-run.

- **DPO / RLHF / GRPO / PPO** — no preference optimization, no reward modeling,
  no policy-gradient stages. Reasons: (1) Qwen3-1.7B is already DPO'd at scale
  by Alibaba; we inherit that alignment through the frozen base + LoRA.
  (2) Comparable sub-2B speech LMs (Mini-Omni, LLaMA-Omni, CSM) ship without
  it. (3) Our quality bottleneck at this scale is capability, not style, and
  DPO fixes style; it can't make a small model smarter. (4) Cost / week of
  timeline is better spent on more synthetic data (Stage 3 → 60h) or more
  S8 distillation traces.
- **Reward model training** — same reasons.
- **Multilingual** — paper claim is English-only, single assistant voice.
  Future work, not this paper.
- **Tool use / function calling** — not the scope of a speech-LM paper.

---

## 1. Contribution Inventory

### Tier 1 — Safe, paper survives without any single experiment succeeding

**S1. Smallest published edge-deployable full-duplex speech LM.** 1.8B, runs in <8GB at INT4, GGUF/llama.cpp on Mac. Comparison: Moshi 7B (data-center only), Mini-Omni ~1B (half-duplex), CSM 1B (TTS).

**S2. Mamba-into-Qwen3 surgery recipe.** Insert 4 Mamba-2 blocks at positions 6/13/20/27. Zero-init residual gates, distill back to base Qwen3 logits on 50M tokens. ≤5% PPL loss, linear-time inference scaling.

**S3. Multi-stream LM format for SNAC hierarchical RVQ.** Delay pattern + interleaving for SNAC's 12/24/48 Hz hierarchy. Different from Moshi's flat-RVQ approach.

### Tier 2 — Valuable if they land

**S4. Resolution-aware depth decoder.** 2-layer Transformer, ~12M params, predicts all 3 SNAC resolutions per step with cross-resolution attention.

**S5. Constrained multi-stream decoding.** Logit-processor layer guaranteeing valid token streams + delay-pattern correctness.

### Tier 3 — Medium risk

**S6. Adaptive reasoning routing via Qwen3 /think.** Gate learns *when* to invoke Qwen3's native reasoning mode during streaming speech. Lower risk than the original from-scratch cognition-gate design because the reasoning capability is inherited from Qwen3 pretraining.

### Tier 4 — Big bet (B)

**S7. On-device continual personalization with catastrophic-forgetting prevention.** First speech LM that adapts to the user during deployment without losing general capability. Personal LoRA (rank-8) updated via online fine-tuning + EWC regularization + replay buffer. Hot-swappable / exportable. Eval introduces new metrics for online speech-LM adaptation.

### Tier 5 — Moonshot (E)

**S8. Distillation recipe closing the Moshi-7B quality gap at 25% of the size.** Two-teacher distillation:
- Qwen3-32B for text reasoning quality (cheap, via OpenRouter)
- Moshi-7B for STS-specific behaviors (if compute available)

Plus extended synthetic corpus (30h → 150h) for better dialogue coverage. Claim: "75–85% of Moshi quality at 25% of params, trained for $200." If it lands, this is the most cited result in the paper.

---

## 2. Architecture (unchanged from v1 of this plan)

### 2.1 Parameter spec

```
─────────────────────────────────────────────────────────────────
                       Vanitas-SLM
─────────────────────────────────────────────────────────────────
  Component                          Params      Source      Trained?
─────────────────────────────────────────────────────────────────
  SNAC codec (24 kHz, 3 levels)       ~20M       frozen      no
  Qwen3-1.7B backbone                 1.7B       4-bit fr.   no
  + 4 Mamba-2 layers (S2)             ~120M      yours       yes
  + Audio-token embeddings            ~8M        yours       yes
  + Resolution-aware depth dec (S4)   ~12M       yours       yes
  + Logit-constraint module (S5)      ~0M        rule        no
  + Reasoning-routing gate (S6)       ~1M        yours       yes
  + General LoRA on Qwen3 (rank-32)   ~20M       yours       yes
  + Personal LoRA (rank-8, per-user)  ~5M        yours       online
─────────────────────────────────────────────────────────────────
  Total inference params              ~1.89B
  Total trainable params              ~166M
  INT4 deployment size                ~1.0 GB
─────────────────────────────────────────────────────────────────
```

### 2.2 Mamba surgery diagram

```
Qwen3-1.7B has 28 transformer layers. We replace 4 of them:

  L0─L1─L2─L3─L4─L5─[M]─L7─L8─L9─L10─L11─L12─[M]─
  L14─L15─L16─L17─L18─L19─[M]─L21─L22─L23─L24─L25─L26─[M]

  [M] = Mamba-2 block (d_model=2048, d_state=128, expand=2)
       residual output projection zero-initialized

  At training start: behavior ≡ Qwen3-1.7B
  After Stage 1 distillation: SSM signal carried, base behavior preserved
```

### 2.3 Multi-stream data flow

```
─────────────────────────────────────────────────────────────────
user audio  ─► SNAC enc ─► user_codes[3 res × N frames]
agent audio ─► SNAC enc ─► agent_codes[3 res × N frames]

Per 80-ms timestep t, model sees and emits:
  Input column:  [user_l0_t, user_l1_t, user_l2_t,
                  agent_text_{t-2},
                  agent_l0_{t-1}, agent_l1_{t-1}, agent_l2_{t-1}]
  Output column: [agent_text_t,
                  agent_l0_t, agent_l1_t, agent_l2_t,
                  p_think_t]

Delay pattern: agent_text leads audio by 2 ticks (160 ms).
Constrained decoding (S5) ensures valid IDs + pattern.
─────────────────────────────────────────────────────────────────
```

### 2.4 Reasoning routing (S6 revised)

```
At each turn boundary:
  1. Backbone forward → hidden h_t
  2. Gate: p_think_t = sigmoid(MLP(h_t))
  3. If p_think_t > τ (default 0.5):
        - Prepend /think tag to next assistant generation
        - Qwen3 emits <think>...</think> in the inner-monologue stream
        - Audio output paused (silence or "let me think" filler)
        - Then resumes with reasoned answer
     Else:
        - Prepend /no_think tag
        - Direct response, sub-200ms TTFA preserved

Target firing rate: 5–15% of turns.
Sparsity loss: KL(observed || 0.10).
```

### 2.5 Continual learning (S7) deployment flow

```
─────────────────────────────────────────────────────────────────
            Vanitas-SLM + Lifelong Adaptation (S7)
─────────────────────────────────────────────────────────────────

  At deployment, every Vanitas instance has:

    ┌──────────────────────────────────────────────┐
    │  Frozen base (Qwen3 + Mamba + depth dec)     │
    │       +                                       │
    │  General LoRA (shipped by us, rank-32)        │
    │       +                                       │
    │  Personal LoRA (rank-8, starts at zero)       │ ← grows with user
    │       +                                       │
    │  Episodic memory: 1k recent turns             │
    │       +                                       │
    │  Replay buffer: 5k samples from general data  │
    └──────────────────────────────────────────────┘

  Every N conversations (or during idle/charging):
    1. Sample recent turns + replay buffer slice
    2. ~500 steps of LoRA fine-tune on personal LoRA only
    3. EWC penalty against general LoRA's Fisher info
       (prevents catastrophic forgetting)
    4. Snapshot if held-out general-task quality preserved

  At inference:
    - Personal LoRA stacks on top of general LoRA (sum of deltas)
    - Hot-swappable: user can reset, share, export, A/B test
─────────────────────────────────────────────────────────────────
```

### 2.6 Latency budget

| Component | ms (RTX 4090, INT4) |
|---|---|
| SNAC streaming encode (pipelined) | ~30 |
| Backbone prefill, first token | ~70 |
| Depth decoder (3 resolutions in one pass) | ~50 |
| Reasoning gate (when off, free; when on, +200–500ms tail) | 0 / variable |
| SNAC decode | ~30 |
| **TTFA (simple turn, no thinking)** | **~180 ms** |
| **TTFA (gate fires, /think mode)** | 380–680 ms |

Mac mini M3 target (via GGUF/llama.cpp): ~220 ms simple, 420–720 ms thinking.

---

## 3. Training Curriculum — 9 Stages (with B+E additions)

### Stage 1 — Mamba surgery + distillation (3 days, $80)

1. Load Qwen3-1.7B via Unsloth (4-bit QLoRA mode).
2. Expand tokenizer with ~6k SNAC tokens.
3. Insert 4 Mamba-2 blocks at positions 6/13/20/27, zero-init residuals.
4. Distill against frozen base Qwen3 logits on 50M tokens (OpenWebText subset).
5. AdamW, LR 5e-5, cosine, batch 16, grad accum to 256.

**Exit:** WikiText-103 PPL within 5% of base Qwen3-1.7B.

### Stage 2 — SNAC token alignment (3 days, $80)

1. Generate ~200h Kokoro TTS audio from Common Voice transcripts (local 4090).
2. Encode all with SNAC → `(text, snac_codes)` tuples.
3. Train ASR (`<text>...<audio>`) + TTS (`<audio>...<text>`) dual format.
4. Loss: NLL on text + NLL on SNAC tokens.
5. Mamba + LoRA + audio embeddings + depth decoder train; Qwen3 4-bit frozen.

**Exit:** LibriSpeech test-clean WER ≤ 18%. Reconstruction UTMOS ≥ 3.3.

### Stage 3 — Synthetic duplex corpus (1 week, $15)

1. Sample 5k UltraChat-200k prompts (short conversational).
2. Expand via Qwen2.5-7B (OpenRouter) into duplex scripts with `<user>`/`<assistant>`/`<backchannel>`/`<overlap>`.
3. TTS via Kokoro-82M with 2 distinct voices.
4. Mix stereo (user L, assistant R) with: 200–700ms gaps, 10% overlaps, 5% backchannels.
5. SNAC-encode both channels.

**Exit:** 30h of audio passes 50-sample listener screen (≥80% intelligible, ≥60% natural dialogue).

### Stage 4 — Multi-stream duplex SFT (4 days, $100)

1. Load Stage 2 checkpoint.
2. Switch to 3-stream input/output with delay pattern.
3. Loss weights: agent_text=1.0, agent_codes (3 res sum)=0.5, user_codes=0.0.
4. AdamW, LR 1e-4, cosine, batch 8, grad accum to 128.

**Exit:** Generated-speech WER ≤ 25%, UTMOS ≥ 3.2, RTF ≤ 1.0.

### 🎯 **Week 6 Decision Gate**

After Stage 4 completion, evaluate base quality and decide big-bet path:

| Stage 4 Quality | Recommended path | Why |
|---|---|---|
| WER ≤ 20% AND UTMOS ≥ 3.4 (strong base) | **Commit to S7 (B)** — base is good enough that personalization will be impressive | continual learning on a strong base demonstrates the recipe |
| WER 20–25% OR UTMOS 3.2–3.4 (acceptable) | **Pursue S8 (E)** distillation to lift quality first, then S7 if time | weak base + personalization = unimpressive demo |
| WER > 25% OR UTMOS < 3.2 (below threshold) | **Drop both big bets** — ship S1–S6 only | base needs more work; ambition won't save it |

### Stage 4.5 (B-track) — Continual learning infrastructure (4 days, $30)

Only run if committing to S7.

1. Build "synthetic user" benchmark: 50 simulated users with distinct vocabulary preferences + voice characteristics (synth speaker IDs).
2. Implement personal-LoRA hot-swap and online update mechanism.
3. Implement EWC computation: Fisher info for general LoRA computed once after Stage 4.
4. Implement replay buffer + general-task held-out eval.

**Exit:** Infrastructure runs end-to-end on a single sample user without crashing. No quality target yet.

### Stage 4.6 (E-track) — Teacher distillation (5 days, $50)

Only run if committing to S8.

1. Run Qwen3-32B on 10k reasoning prompts via OpenRouter (~$30) → collect logits/responses.
2. *Optional, if budget allows:* run Moshi-7B on 5k STS prompts via rented A100 ($20) → collect codec streams.
3. Add distillation loss to Stage 4 training: KL(student || Qwen3-32B teacher) for text, audio-token NLL matching for Moshi-trace (if available).
4. Continue training for ~2 days with the combined loss.

**Exit:** Reasoning eval (GSM8K-spoken or MMLU-spoken proxy) improves ≥ 5 pts over Stage 4 baseline.

### Stage 4.7 (E-track) — Extended synthetic corpus (4 days, $60)

Only run if committing to S8.

1. Generate additional 120h of synthetic dialogue (total now 150h) covering more topics: reasoning, multi-turn, longer contexts.
2. Continue Stage 4 training on combined 150h with distillation loss active.

**Exit:** Generated WER ≤ 18% (significant lift from 25%).

### Stage 5 — Reasoning routing gate fine-tune (2 days, $30)

1. **5a (supervised):** Re-generate 5h of corpus with explicit `/think` mode labels on reasoning-heavy turns. Train gate with frame-level BCE.
2. **5b (semi-supervised):** Drop labels, add sparsity KL + hard-concrete + temperature annealing.
3. At inference, when `p_think > τ`, prepend `/think` to next assistant turn.

**Exit:** Firing rate ∈ [3%, 25%] AND reasoning subset accuracy uplift ≥ 2 pts. Failure path: drop gate, paper unaffected.

### Stage 5.5 (B-track) — Online personalization training + eval (1 week, $40)

Only run if committing to S7.

1. For each of 50 simulated users:
   - 20 sessions of interaction
   - Personal LoRA updated after each session via online fine-tune + EWC
2. Track two curves:
   - **Personalization:** task-specific accuracy on user's vocabulary/style over sessions
   - **Forgetting:** general-task quality on held-out benchmark
3. Report uplift on user-specific metrics + preservation of general capability.

**Exit:** ≥10% improvement on personal tasks averaged across 50 users WITH ≤2 pt drop on general benchmarks.

### Stage 6 — Eval + latency optimization + GGUF (3 days, $40)

- INT4 quantization (Unsloth dynamic)
- `torch.compile` + Flash-Attn 2
- KV cache + constant Mamba state
- Speculative decoding on text stream (Qwen3-0.6B draft)
- GGUF export for Mac

**Exit:** TTFA ≤ 200 ms on 4090. GGUF loads + generates on Mac mini M3.

---

## Budget Summary (Final)

| Stage | Cost |
|---|---|
| 1. Mamba surgery + distill | $80 |
| 2. SNAC alignment | $80 |
| 3. Synthetic data gen | $15 |
| 4. Duplex SFT | $100 |
| **4.5. Continual-learning infra (B)** | **$30** |
| **4.6. Teacher distillation (E)** | **$50** |
| **4.7. Extended corpus (E)** | **$60** |
| 5. Reasoning routing gate | $30 |
| **5.5. Online personalization (B)** | **$40** |
| 6. Eval + optimization | $40 |
| **Subtotal (all stages)** | **$525** |
| Buffer (2× re-runs) | $200 |
| **Realistic ceiling** | **$725** |
| Cash floor (with free tiers stacked) | **~$380** |

---

## 4. Evaluation Suite

### 4.1 Headline table

| Model | Params | Duplex | Edge? | TTFA | WER ↓ | UTMOS ↑ | MTR-Duplex ↑ |
|---|---|---|---|---|---|---|---|
| Moshi | 7B | ✓ | ✗ | ~200ms (H100) | ~10% | ~3.8 | 3.13 |
| Freeze-Omni | ~7B | ✓ | ✗ | ~600ms | — | — | 3.48 |
| Mini-Omni | ~1B | ✗ | ~ | ~600ms | — | ~3.5 | — |
| CSM-1B | 1B | ✗ (TTS) | ~ | — | — | ~4.0 | — |
| **Vanitas-SLM (ours)** | **1.8B** | **✓** | **✓** | **≤200ms** | **≤18% (with E) / ≤25% (without)** | **≥3.4 / ≥3.2** | **≥3.0** |

### 4.2 S2 ablation: Mamba surgery

| Backbone | Trainable | PPL | TTFA @ 1k ctx | Throughput @ 10k ctx |
|---|---|---|---|---|
| Qwen3-1.7B base | LoRA only | baseline | baseline | baseline |
| + 2 Mamba | +60M | +2% | similar | +25% |
| **+ 4 Mamba (ours)** | **+120M** | **+5%** | **similar** | **+50%** |
| + 8 Mamba | +240M | +12% | — | — |

### 4.3 S3+S4 ablation: Multi-stream format + depth decoder

| Variant | WER ↓ | UTMOS ↑ | TTFA |
|---|---|---|---|
| Flattened RVQ (sequential) | baseline | baseline | ~280ms |
| Parallel 3-level, no cross-attn | better | similar | ~200ms |
| **Resolution-aware depth dec (ours)** | **best** | **best** | **~180ms** |

### 4.4 S6 ablation: Reasoning routing

| Variant | Fire rate | Reasoning acc ↑ | Simple TTFA | Reasoning TTFA |
|---|---|---|---|---|
| Gate off | 0% | baseline | baseline | n/a |
| Gate τ=0.7 sparse | ~5% | +3 pts | +0 ms | +400 ms |
| Gate τ=0.5 default | ~12% | +6 pts | +5 ms | +450 ms |
| Gate τ=0.3 eager | ~25% | +6 pts (plateau) | +20 ms | +500 ms |

### 4.5 S7 ablation: Continual learning (B)

| Variant | Personal acc ↑ over 20 sessions | General acc preservation | LoRA size |
|---|---|---|---|
| No adaptation | 0% baseline | 100% baseline | 0 MB |
| Personal LoRA only, no EWC | +18% | -8% (forgetting!) | +20 MB |
| **Personal LoRA + EWC (ours)** | **+12%** | **-2%** | **+20 MB** |
| Personal LoRA + replay only | +10% | -4% | +20 MB |

### 4.6 S8 ablation: Distillation (E)

| Setup | WER ↓ | Reasoning acc | TTFA | Distance to Moshi |
|---|---|---|---|---|
| Stage 4 baseline (no distillation) | 25% | baseline | 180ms | 60% of quality |
| + Qwen3-32B distillation | 20% | +8 pts | 180ms | 75% of quality |
| **+ Moshi STS distillation (ours full)** | **18%** | **+10 pts** | **180ms** | **85% of quality** |

### 4.7 Latency decomposition (S1)

Detailed per-component table from §2.6, plus measurement on 4090, A100, Mac mini M3.

### 4.8 Benchmarks (full)

- LibriSpeech test-clean (ASR WER)
- Whisper-judged WER on generated speech (TTS quality)
- UTMOS / DNSMOS (voice naturalness)
- Full-Duplex-Bench v1.5 (turn-taking, barge-in, backchannel)
- MTR-DuplexBench (multi-round dialogue)
- FD-Bench (full pipeline)
- GSM8K-spoken / MMLU-spoken (reasoning under speech, for S6 + S8)
- AlpacaEval-spoken (assistant quality)
- *Custom*: 50-user personalization benchmark (S7)
- *Custom*: Moshi-comparison test set (S8)

---

## 5. Repo Layout

Archive everything currently on `main` to `legacy/v1` branch. New layout:

```
vanitas/
  codec/snac_wrapper.py
  tokenizer/expand_vocab.py
  model/
    backbone.py                # Qwen3 + Mamba surgery
    mamba_block.py
    depth_decoder.py           # S4
    cognition_gate.py          # S6 (reasoning routing)
    constraints.py             # S5
    vanitas_slm.py             # multi-stream forward
    personal_lora.py           # S7: per-user adapters
  data/
    synth_duplex.py            # Stage 3 corpus builder
    snac_encode.py
    streams.py                 # delay-pattern interleaving
  distillation/                # S8
    teacher_qwen3_32b.py       # OpenRouter pipeline
    teacher_moshi.py           # optional A100 inference
    distill_loss.py
  continual/                   # S7
    online_lora.py
    ewc.py
    replay_buffer.py
    user_simulator.py
  training/
    stage1_mamba_distill.py
    stage2_snac_align.py
    stage3_duplex_sft.py
    stage4_5_continual_infra.py
    stage4_6_teacher_distill.py
    stage4_7_extended_corpus.py
    stage5_reasoning_gate.py
    stage5_5_personalization.py
    common.py
  eval/
    latency.py
    fd_bench.py
    mtr_bench.py
    utmos.py
    spoken_reasoning.py
    personalization_bench.py   # S7
    moshi_comparison.py        # S8
  inference/
    streaming_server.py
    speculative.py
    constrained_decode.py
    gguf_export.py
    client_demo.py
  audio/{capture,playback}.py  # kept from v1
  tests/
```

---

## 6. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Mamba surgery degrades Qwen3 | Med | High | Zero-init, distill, hard exit, fallback 2 Mamba |
| 2 | SNAC token learning hard for LoRA-only | Med | High | If WER >35%, unfreeze last 4 Qwen3 layers |
| 3 | Synthetic data robotic | Med | Med | 2+ voices, randomized timing, optional DailyTalk mix |
| 4 | Reasoning gate (S6) fails | Med | Low | Sparsity KL + curriculum + documented fallback |
| 5 | TTFA > 250ms | Low | Med | Profile, INT4 + Flash-Attn + torch.compile |
| 6 | Quality below Mini-Omni after Stage 4 | Med | High | Main scientific risk. E (distillation) is the recovery path. |
| 7 | **S7 catastrophic forgetting unsolved** | **Med** | **High** | EWC + replay + held-out eval. If forgetting >5%, document as negative result. |
| 8 | **S7 personalization plateaus / no measurable gain** | **Med** | **Med** | Use stronger user simulator. Report negative result if needed. |
| 9 | **S8 Qwen3-32B distillation doesn't transfer** | **Med** | **Med** | Try smaller teacher (Qwen3-8B). Report negative result. |
| 10 | **S8 Moshi inference costs blow budget** | **Med** | **Med** | Skip Moshi-teacher, stay with Qwen3-32B only. |
| 11 | Unsloth doesn't support Mamba in QLoRA | Low | Med | Fall back to standard PEFT+bitsandbytes |
| 12 | Reviewers reject "pretrained base" | Low | Low | Cite Mini-Omni, LLaMA-Omni, CSM precedent |

---

## 7. Nine-Week Timeline

| Week | Stages | Exit |
|---|---|---|
| 1 | Repo scaffold + Stage 1 | PPL within 5% of base |
| 2 | Stage 2a: TTS data gen + SNAC encode | 200h encoded |
| 3 | Stage 2b: train SNAC alignment | WER ≤ 18%, UTMOS ≥ 3.3 |
| 4 | Stage 3: synthetic duplex corpus | 30h passes listener screen |
| 5 | Stage 4: duplex SFT | WER ≤ 25%, UTMOS ≥ 3.2, RTF ≤ 1.0 |
| **6** | **🎯 Decision Gate + chosen big-bet track** | **Quality assessment + commit to B/E/both/neither** |
| 7 | Stage 4.5 + 4.6 (parallel, depending on track) | Infrastructure + teacher distillation done |
| 8 | Stage 4.7 + 5 + 5.5 (parallel) | Extended corpus + reasoning gate + personalization eval |
| 9 | Stage 6 + paper draft | Submission-ready |

Daily commitment: 3–4 focused hours. Weekly check-ins each Friday.

---

## 8. Paper Outline (8 pages + appendix)

1. **Introduction.** Position: full-duplex speech LMs exist (Moshi, Freeze-Omni) but require data-center hardware. We present a 1.8B alternative that runs on consumer devices, adapts to users, and approaches frontier quality via distillation.
2. **Related work.** Moshi, Mini-Omni, CSM, Spirit-LM, Mamba/Mamba-2, Jamba/Zamba, SNAC, EWC, online continual learning.
3. **Method.**
   - 3.1 Hybrid Mamba-Qwen3 backbone (S2)
   - 3.2 Multi-stream format for hierarchical RVQ (S3)
   - 3.3 Resolution-aware depth decoder (S4)
   - 3.4 Constrained multi-stream decoding (S5)
   - 3.5 Adaptive reasoning routing (S6)
   - 3.6 **On-device continual personalization (S7) — if landed**
   - 3.7 **Two-teacher distillation (S8) — if landed**
4. **Experiments.** All ablation tables from §4.
5. **Discussion.** Limits: English-only, single assistant voice, no tool use. Open questions: scaling, multilingual, federated learning.
6. **Conclusion.** A reproducible recipe for edge-deployable, continually-adapting, near-frontier speech LMs.

**Release artifacts:**
- Code: `github.com/dhanyamd/vanitas-slm` (Apache-2)
- Weights: `huggingface.co/<you>/vanitas-slm-1.8b` (Apache-2)
- Dataset: `huggingface.co/datasets/<you>/vanitas-duplex-150h` (CC-BY-4.0)
- Demo video: Mac-mini live deployment, ≤3 min
- Personalization benchmark: open release of the 50-user simulator

---

## 9. What you give the world if S7 + S8 both land

1. **A working full-duplex speech LM small enough to run on a Mac.**
2. **A reproducible Mamba-into-Transformer surgery recipe.**
3. **A multi-stream format for hierarchical RVQ codecs.**
4. **A resolution-aware decoder.**
5. **The first published demonstration of adaptive reasoning during streaming speech.**
6. **The first speech LM that adapts to users on-device without forgetting.**
7. **A distillation recipe that closes most of the gap to frontier speech LMs at 25% of the size.**
8. **A 150h synthetic duplex assistant dataset.**
9. **A complete training pipeline reproducible for under $750** by other independent researchers.

If even half of these land, this is **a paper that gets remembered**, not just published.

---

## 10. The "do not skip" checklist

- [ ] **Pre-Stage 1:** SNAC round-trip reconstructs intelligible 10-utterance sample
- [ ] **Stage 1:** Hybrid PPL within 5% of base Qwen3
- [ ] **Stage 2:** ASR WER ≤ 18% on LibriSpeech test-clean
- [ ] **Stage 2:** Reconstruction UTMOS ≥ 3.3
- [ ] **Stage 3:** 30h corpus passes listener screen
- [ ] **Stage 4:** Generated WER ≤ 25%, UTMOS ≥ 3.2, RTF ≤ 1.0
- [ ] **🎯 Week 6 Decision Gate:** Commit to B / E / both / neither based on Stage 4 quality
- [ ] **Stage 4.5 (B):** Infrastructure runs end-to-end on one user
- [ ] **Stage 4.6 (E):** Reasoning eval +5 pts over Stage 4
- [ ] **Stage 4.7 (E):** Generated WER ≤ 18% with distillation
- [ ] **Stage 5:** Gate fires 3–25% AND reasoning uplift ≥ 2 pts (or drop)
- [ ] **Stage 5.5 (B):** Personal acc +10% across 50 users WITH general -≤2 pts
- [ ] **Stage 6:** TTFA ≤ 200 ms on 4090, GGUF runs on Mac mini M3
- [ ] **Paper readiness:** S1–S5 tables filled, S6 + S7 + S8 either landed or honestly reported

---

## 11. Outcome probability distribution

| Outcome | Probability |
|---|---|
| Dream: S1–S5 + S6 + S7 + S8 all land | ~8% |
| Strong: S1–S5 + S6 + (S7 OR S8) lands | ~30% |
| Solid: S1–S5 + (S6 OR S7 OR S8) lands | ~45% |
| Fallback: Only S1–S5 land | ~15% |
| Catastrophic: Stage 1 or 2 doesn't work | ~2% |

**93% chance of publishable paper. 38% chance of strong/dream paper.**

---

*Plan locked. Next step: archive v1, scaffold the new repo structure, start Stage 1.*
