"""Vocab expansion for Qwen3 tokenizer.

Adds SNAC audio tokens and special control tags to the Qwen3 vocabulary.

Token layout (deterministic, must stay stable across stages):

  Per-level SNAC codes:
      <snac_l0_0>, <snac_l0_1>, ..., <snac_l0_{N0-1}>
      <snac_l1_0>, ..., <snac_l1_{N1-1}>
      <snac_l2_0>, ..., <snac_l2_{N2-1}>
  where N0 == N1 == N2 == codebook_size (default 4096 for snac_24khz).

  Special control tags (stream framing + cognition gate + dialog markers):
      <user_audio>, </user_audio>
      <agent_audio>, </agent_audio>
      <user_text>,  </user_text>      # optional, used when transcripts exist
      <agent_text>, </agent_text>
      <think>, </think>               # inner-monologue / reasoning span
      <silence>                       # explicit silent audio frame
      <backchannel>, </backchannel>   # short acknowledgement marker

The new tokens are appended to Qwen3's existing vocab (do not insert in
the middle — that would invalidate every prior token id).

The result is a HuggingFace tokenizer object plus the bookkeeping needed by
``vanitas.v2.model.backbone`` to resize the model's embedding + lm_head.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from transformers import AutoTokenizer, PreTrainedTokenizerBase

QWEN3_MODEL_ID = "Qwen/Qwen3-1.7B"

# Special tags. Order is part of the saved artifact contract — do not reshuffle.
SPECIAL_TAGS: tuple[str, ...] = (
    "<user_audio>", "</user_audio>",
    "<agent_audio>", "</agent_audio>",
    "<user_text>", "</user_text>",
    "<agent_text>", "</agent_text>",
    "<think>", "</think>",
    "<silence>",
    "<backchannel>", "</backchannel>",
)


@dataclass
class VocabExpansion:
    """Records of what was added, so the model side can resize correctly."""

    base_vocab_size: int
    added_tokens: list[str]
    snac_codebook_sizes: tuple[int, int, int]
    special_tags: tuple[str, ...] = field(default_factory=lambda: SPECIAL_TAGS)

    @property
    def new_vocab_size(self) -> int:
        return self.base_vocab_size + len(self.added_tokens)

    @property
    def total_added(self) -> int:
        return len(self.added_tokens)

    def token_id(self, tokenizer: PreTrainedTokenizerBase, token: str) -> int:
        """Convenience: look up an added token by string."""
        ids = tokenizer.convert_tokens_to_ids([token])
        return ids[0]

    def snac_token(self, level: int, code: int) -> str:
        assert 0 <= level < 3
        assert 0 <= code < self.snac_codebook_sizes[level]
        return f"<snac_l{level}_{code}>"


def build_snac_token_names(codebook_sizes: Sequence[int]) -> list[str]:
    """Generate the deterministic token-name list for SNAC codes."""
    names: list[str] = []
    for level, size in enumerate(codebook_sizes):
        for code in range(size):
            names.append(f"<snac_l{level}_{code}>")
    return names


def expand_qwen3_tokenizer(
    base_model_id: str = QWEN3_MODEL_ID,
    snac_codebook_sizes: tuple[int, int, int] = (4096, 4096, 4096),
    save_dir: str | Path | None = None,
) -> tuple[PreTrainedTokenizerBase, VocabExpansion]:
    """Return a Qwen3 tokenizer with SNAC + special tags appended.

    Args:
        base_model_id: HF id of the base Qwen3 model whose tokenizer we extend.
        snac_codebook_sizes: per-level codebook sizes (typically (4096,)*3).
        save_dir: if provided, writes the expanded tokenizer + a JSON manifest
            of the expansion so later stages can re-load deterministically.

    Returns:
        (tokenizer, expansion) where ``expansion`` records the added tokens and
        their bookkeeping (used by the model to resize embed/lm_head and by
        downstream stages for token-id lookups).
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_vocab_size = len(tokenizer)

    snac_tokens = build_snac_token_names(snac_codebook_sizes)
    added_tokens: list[str] = list(SPECIAL_TAGS) + snac_tokens

    # ``add_tokens`` is idempotent: tokens already present are no-ops. It also
    # appends rather than inserts, so ids 0..base_vocab_size-1 remain stable.
    num_actually_added = tokenizer.add_tokens(added_tokens, special_tokens=True)

    expansion = VocabExpansion(
        base_vocab_size=base_vocab_size,
        added_tokens=added_tokens,
        snac_codebook_sizes=tuple(snac_codebook_sizes),  # type: ignore[arg-type]
    )

    if num_actually_added != len(added_tokens):
        # Most likely cause: re-running expand on an already-expanded tokenizer.
        # Not fatal, but the caller should know so they don't double-count later.
        print(
            f"[expand_vocab] {num_actually_added}/{len(added_tokens)} tokens were "
            f"actually new; the rest were already in the tokenizer."
        )

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(save_dir)

        # Save a manifest so stage 1 (and later stages) can validate alignment.
        import json
        manifest = {
            "base_model_id": base_model_id,
            "base_vocab_size": base_vocab_size,
            "snac_codebook_sizes": list(snac_codebook_sizes),
            "special_tags": list(SPECIAL_TAGS),
            "new_vocab_size": expansion.new_vocab_size,
            "total_added": expansion.total_added,
        }
        (save_dir / "vanitas_vocab_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )

    return tokenizer, expansion


def _self_test() -> int:
    """Quick smoke test you can run directly. Does not need a GPU."""
    print(f"Loading Qwen3 tokenizer ({QWEN3_MODEL_ID})...")
    tokenizer, expansion = expand_qwen3_tokenizer()

    print(f"  Base vocab size:  {expansion.base_vocab_size:,}")
    print(f"  Added:            {expansion.total_added:,}")
    print(f"  New vocab size:   {expansion.new_vocab_size:,}")
    print(f"  Special tags:     {len(SPECIAL_TAGS)}")
    print(f"  SNAC tokens:      {sum(expansion.snac_codebook_sizes):,}")

    # Sanity: special tags must round-trip through the tokenizer
    for tag in SPECIAL_TAGS:
        ids = tokenizer.encode(tag, add_special_tokens=False)
        assert len(ids) == 1, (
            f"Special tag {tag!r} did not encode to a single token id ({ids}). "
            "The tokenizer split it — something is wrong with add_tokens."
        )

    # Sanity: a SNAC token must also be one token
    sample = expansion.snac_token(level=2, code=42)
    ids = tokenizer.encode(sample, add_special_tokens=False)
    assert len(ids) == 1, f"SNAC token {sample!r} did not encode as single id"

    # Sanity: english text still tokenizes the same as before
    sentence = "Hello, this is a test."
    ids_before = AutoTokenizer.from_pretrained(QWEN3_MODEL_ID).encode(sentence)
    ids_after = tokenizer.encode(sentence)
    assert ids_before == ids_after, (
        "English tokenization changed after adding tokens. Something went wrong."
    )

    print("\n✓ All vocab-expansion self-tests passed.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
