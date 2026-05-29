"""Dependency-free text embeddings for Vanitas memory alignment.

This is not a replacement for a trained text encoder. It is a deterministic
feature-hashing encoder that preserves lexical overlap, unlike one-random-vector
per text hashing. That makes local/offline memory training and tests meaningful
without adding a heavy dependency to Modal.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import torch

TOKEN_RE = re.compile(r"[a-z0-9]+")


def _signed_hash(feature: str, embedding_dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    raw = int.from_bytes(digest, byteorder="little", signed=False)
    index = raw % embedding_dim
    sign = 1.0 if ((raw >> 63) & 1) == 0 else -1.0
    return index, sign


def lexical_text_embedding(text: str, embedding_dim: int = 512) -> torch.Tensor:
    """Encode text with normalized signed feature hashing.

    Similar strings share token and character-ngram features, so cosine
    similarity has a useful lexical signal. This keeps the training pipeline
    dependency-free while avoiding the previous semantically-useless random
    vector per document.
    """
    vector = np.zeros(embedding_dim, dtype=np.float32)
    text_norm = text.lower()
    tokens = TOKEN_RE.findall(text_norm)

    if not tokens:
        return torch.from_numpy(vector)

    for token in tokens:
        # Word feature.
        idx, sign = _signed_hash(f"w:{token}", embedding_dim)
        vector[idx] += sign

        # Prefix/suffix features help related durations, ids, and variants.
        for size in (3, 4):
            if len(token) >= size:
                idx, sign = _signed_hash(f"p{size}:{token[:size]}", embedding_dim)
                vector[idx] += 0.5 * sign
                idx, sign = _signed_hash(f"s{size}:{token[-size:]}", embedding_dim)
                vector[idx] += 0.5 * sign

        # Character ngrams preserve partial lexical similarity.
        padded = f"^{token}$"
        for n in (3, 4, 5):
            if len(padded) >= n:
                for start in range(len(padded) - n + 1):
                    idx, sign = _signed_hash(f"c{n}:{padded[start:start+n]}", embedding_dim)
                    vector[idx] += 0.25 * sign

    # Add simple length/rhythm buckets so generic dialogue metadata is not
    # entirely collapsed when real transcripts are unavailable.
    token_count_bucket = min(31, len(tokens))
    idx, sign = _signed_hash(f"len:{token_count_bucket}", embedding_dim)
    vector[idx] += 0.5 * sign

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return torch.from_numpy(vector)
