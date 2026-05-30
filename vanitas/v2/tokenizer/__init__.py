"""Tokenizer vocab expansion utilities.

Adds SNAC audio tokens to Qwen3's tokenizer in a deterministic layout:
  - Special tags: <user_audio>, </user_audio>, <agent_audio>, </agent_audio>,
                  <think>, </think>, <silence>, <backchannel>
  - One token per SNAC codebook entry per resolution level, namespaced by level.

Resize Qwen3's embedding and lm_head; initialize new rows N(0, 0.02).
"""
