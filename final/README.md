# The Binding Bottleneck — Core Code

Evaluation and analysis code for the paper *"The Binding Bottleneck: Why
Audio-Language Models Hear but Fail to Reason."* This folder holds only the code
that produced the results in the paper; exploratory work from earlier directions is
not included.

## Files
- `sakura_binding_eval.py` — the full evaluation and analysis script. Self-contained;
  each experiment is a separate entry point (a Modal function).
- `paper_figures.py` — generates the two figures in the paper (method diagram and
  results bar chart).

## Models and benchmarks
Frozen, open-weight models (Qwen2-Audio-7B-Instruct, Audio-Flamingo-3 / -Next,
Qwen2-VL) over public benchmarks (SAKURA, MMAR, GQA). No weights are trained in any
test-time condition. Scoring is deterministic exact match on the selected
multiple-choice letter, with no LLM judge.

## Decoding
- Base: greedy, one pass.
- Self-consistency (SC): `K=5` samples (temperature `0.7`, top-p `0.5`, top-k `20`),
  majority vote over parsed option letters, ties broken by the greedy answer.

## Paper result to function
| Result | Function |
| --- | --- |
| Base and SC (SAKURA) | `qwen_baseline`, `qwen_base_sc` |
| No-audio control | `qwen_noaudio_sc` |
| Oracle perception injection | `qwen_tool_perceive` |
| Logit-lens localization (L24 to L32) | `qwen_binding_probe` |
| Decoupled cascade upper bound | `qwen_cascade` |
| Elicitation with provided single-hop question | `qwen_decomp_sc` |
| Self-derived elicitation (stress test) | `qwen_bind_selfask` |
| Cross-model (AF-Next) | `afnext_base_sc`, `afnext_decomp` |
| MMAR base/SC and reasoning-layer breakdown | `mmar_base_sc`, `mmar_breakdown` |
| Vision (GQA) control | `gqa_inspect`, `vl_binding_gate`, `vl_decomp` |
| Binding-aware GRPO (negative result) | `qwen_grpo_smoke` |
| Structured-CoT SFT (negative result) | `qwen_structured_data`, `qwen_structured_eval` |

## Running
Entry points run via [Modal](https://modal.com), e.g.:

    modal run sakura_binding_eval.py::qwen_base_sc

Benchmarks are pulled from their public sources at run time.
