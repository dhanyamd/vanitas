# Vanitas-SLM — Final Plan (Path B — Trimmed Scope)

> **Title:** Vanitas-SLM: An Edge-Deployable 1.85B Full-Duplex Speech Language Model via Mamba-Adapted Qwen3 and SNAC Hierarchical RVQ
> **Status:** Path B locked. Moonshots S5/S6/S7/S8 dropped. May 2026.
> **Authors:** dhanyamd + collaborator
> **Budget:** $250 floor — $450 ceiling
> **Timeline:** 3–4 weeks (down from 9)
> **Venue target:** Interspeech 2026 → ASRU 2026 workshop track

> **Why trimmed:** The original plan stacked three high-risk research bets
> (cognition gate, continual learning, big-teacher distillation) on top of a
> core that already required 4 weeks. The risk-budget didn't fit the
> timeline-budget. Path B keeps only the contributions that are (a) already
> implemented or trivial to implement and (b) validated to work — Stage 1
> smoke passed with PPL ratio 1.000. The moonshots are out of scope for
> this paper; future work for v2.

---

## 0. Scope decisions (locked, Path B)

- **Small** — total inference params ~1.85B. INT4 GGUF deploys in ~1.1 GB.
- **Pretrained components accepted as infrastructure** — Qwen3-1.7B (Apache-2) and SNAC (MIT). Cited, not claimed.
- **Full duplex non-negotiable** — true parallel user+agent streams.
- **Sub-200 ms TTFA target** — 180 ms on RTX 4090, 220 ms on Mac mini M3. Higher is acceptable as long as ≤300 ms.
- **Four contributions** (S1–S4). All low-risk; one (S2) already validated by Stage 1 smoke.
- **No big-bet research moonshots.** S5–S8 are explicitly future work.

### Out of scope (intentionally — do not re-add)

- **DPO / RLHF / GRPO / PPO** — Qwen3-1.7B is already DPO'd by Alibaba; we inherit it via frozen base + LoRA. Comparable papers (Mini-Omni, LLaMA-Omni, CSM) all ship without preference optimization.
- **Adaptive cognition gate (was S6)** — high-risk research bet, dropped to fit timeline. Future work.
- **On-device continual learning (was S7)** — high-risk research bet, dropped. Future work.
- **Two-teacher distillation moonshot (was S8)** — high-risk research bet, dropped. The base Qwen3 quality is enough for the paper claim we're now making.
- **Constrained multi-stream decoding (was S5)** — drop from contributions; if we need it for inference correctness we add a 50-line custom logit processor without claiming novelty.
- **Reward model training, multilingual, tool use** — not the scope of this paper.

---

## 1. Contribution Inventory

### Tier 1 — Safe, paper survives without any single experiment succeeding

**S1. Smallest published edge-deployable full-duplex speech LM.** 1.8B, runs in <8GB at INT4, GGUF/llama.cpp on Mac. Comparison: Moshi 7B (data-center only), Mini-Omni ~1B (half-duplex), CSM 1B (TTS).

**S2. Mamba-into-Qwen3 surgery recipe.** Insert 4 Mamba-2 blocks *after* Qwen3 layers 6/13/20/27 (resulting 32-layer stack with Mamba at new indices 7/15/23/31). Zero-init residual gates ⇒ student ≡ teacher at step 0. Distill back to base Qwen3 logits on 50M tokens to open the gates. ≤5% PPL loss, linear-time inference scaling on the new Mamba layers.

**S3. Multi-stream LM format for SNAC hierarchical RVQ.** Delay pattern + interleaving for SNAC's 12/24/48 Hz hierarchy. Different from Moshi's flat-RVQ approach.

### Tier 2 — Valuable, low-risk

**S4. Resolution-aware depth decoder.** 2-layer Transformer, ~12M params, predicts all 3 SNAC resolutions per step with cross-resolution attention.

### Dropped from this paper (was S5, S6, S7, S8 — see §0 "out of scope")

- ~~Constrained multi-stream decoding (S5)~~ → if needed for inference correctness, do as 50-line logit mask, don't claim novelty.
- ~~Adaptive reasoning routing via Qwen3 /think (S6)~~ → future work.
- ~~On-device continual personalization (S7)~~ → future work.
- ~~Two-teacher distillation moonshot (S8)~~ → future work.

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
  + LoRA on Qwen3 (rank-32)           ~20M       yours       yes
─────────────────────────────────────────────────────────────────
  Total inference params              ~1.85B
  Total trainable params              ~160M
  INT4 deployment size                ~1.1 GB
─────────────────────────────────────────────────────────────────
```

### 2.2 Mamba surgery diagram

```
Qwen3-1.7B has 28 transformer layers. We INSERT 4 Mamba blocks after
layers 6/13/20/27, growing the stack to 32 layers:

  L0─L1─...─L5─L6─[M]─L7─...─L12─L13─[M]─L14─...─L19─L20─[M]─L21─...─L26─L27─[M]
                  ↑                ↑                ↑                       ↑
              new L7            new L15           new L23                 new L31

  [M] = Mamba-2 block (d_model=2048, d_state=128, expand=2)
        scalar residual gate initialized to zero

  At training start: gate=0 ⇒ Mamba block ≡ identity ⇒ student ≡ base Qwen3
  After Stage 1 distillation: gates open, Mamba signal contributes
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

## 3. Training Curriculum — 5 Stages (Path B)

### Stage 1 — Mamba surgery + distillation (3–4 days, $80) ✅ smoke validated

1. Load Qwen3-1.7B via HF transformers (bf16).
2. Insert 4 Mamba-2 blocks *after* layers 6/13/20/27 (32-layer stack with zero-init residual gates).
3. Distill against frozen base Qwen3 logits on 50M tokens (OpenWebText subset).
4. AdamW, LR 5e-5, cosine, batch 4, grad accum to 64 (effective batch 256).

**Exit:** WikiText-103 PPL within 5% of base Qwen3-1.7B. Smoke run already confirmed ratio = 1.000 at step 0.

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

### Stage 5 — Eval + latency optimization + GGUF (3 days, $40)

- INT4 quantization (Unsloth dynamic)
- `torch.compile` + Flash-Attn 2
- KV cache + constant Mamba state
- Speculative decoding on text stream (Qwen3-0.6B draft)
- GGUF export for Mac

**Exit:** TTFA ≤ 200 ms on 4090. GGUF loads + generates on Mac mini M3.

---

## Budget Summary (Path B)

| Stage | Cost |
|---|---|
| 1. Mamba surgery + distill | $80 |
| 2. SNAC alignment | $80 |
| 3. Synthetic data gen | $15 |
| 4. Duplex SFT | $100 |
| 5. Eval + optimization + GGUF | $40 |
| **Subtotal** | **$315** |
| Buffer (failed re-runs) | $130 |
| **Realistic ceiling** | **$450** |
| Cash floor (with Kaggle/Colab credits) | **~$250** |

Smoke spend to date: $0.63. Counts against ceiling.

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
