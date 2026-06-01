"""VanitasKVMemory — native, bounded, salience-gated persistent KV memory.

Our own implementation. Inspired by (NOT copied from) Memorizing Transformers
(Wu et al., ICLR 2022, arXiv:2203.08913). Differences: speech-LM domain,
per-user cross-session persistence, no Faiss/ANN (plain torch top-k — our store
is small enough), salience-gated writes + bounded eviction to prevent rot.

Key fact this module demonstrates: at *personal-assistant* scale (hundreds to a
few thousand memories for one user), recall is a single small matmul — sub-
millisecond on GPU, ~1 ms on CPU — and the whole store is a few MB. There is no
meaningful latency-vs-memory tradeoff at this scale; that only appears at
web-scale (millions of entries) RAG.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class VanitasKVMemory:
    """Bounded persistent key/value memory with salience-gated writes.

    Shapes use (n_kv_heads, head_dim) per entry to match grouped-query attention.

    Args:
        max_entries: hard cap on stored memories (bounds latency + RAM).
        n_kv_heads:  number of KV heads in the host attention layer.
        head_dim:    per-head dimension.
        topk:        how many memories to retrieve per query.
        device/dtype: storage device/dtype (fp16 keeps it tiny).
    """

    def __init__(
        self,
        max_entries: int = 1024,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        topk: int = 8,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.max_entries = max_entries
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.topk = topk
        self.device = device
        self.dtype = dtype

        self.K = torch.zeros(0, n_kv_heads, head_dim, device=device, dtype=dtype)
        self.V = torch.zeros(0, n_kv_heads, head_dim, device=device, dtype=dtype)
        # Bookkeeping for usefulness x recency eviction.
        self.timestamp = torch.zeros(0, device=device, dtype=torch.long)
        self.hits = torch.zeros(0, device=device, dtype=torch.long)
        self._t = 0

    # -- size / persistence ------------------------------------------------
    def __len__(self) -> int:
        return self.K.shape[0]

    def memory_bytes(self) -> int:
        return (self.K.numel() + self.V.numel()) * self.K.element_size()

    def save(self, path: str) -> None:
        torch.save(
            {"K": self.K.cpu(), "V": self.V.cpu(),
             "timestamp": self.timestamp.cpu(), "hits": self.hits.cpu(), "t": self._t},
            path,
        )

    def load(self, path: str) -> None:
        d = torch.load(path, map_location=self.device)
        self.K = d["K"].to(self.device, self.dtype)
        self.V = d["V"].to(self.device, self.dtype)
        self.timestamp = d["timestamp"].to(self.device)
        self.hits = d["hits"].to(self.device)
        self._t = d["t"]

    # -- write (no backprop) ----------------------------------------------
    @torch.no_grad()
    def write(self, k: torch.Tensor, v: torch.Tensor, salience: float = 1.0,
              salience_threshold: float = 0.5) -> bool:
        """Append salient key/value entries. Returns True if written.

        Salience gating: trivial turns (salience below threshold) are forgotten
        immediately — most turns are never stored, like human memory.
        k, v: (n_new, n_kv_heads, head_dim).
        """
        if salience < salience_threshold:
            return False
        k = k.detach().to(self.device, self.dtype)
        v = v.detach().to(self.device, self.dtype)
        n_new = k.shape[0]
        self.K = torch.cat([self.K, k], dim=0)
        self.V = torch.cat([self.V, v], dim=0)
        ts = torch.full((n_new,), self._t, device=self.device, dtype=torch.long)
        self.timestamp = torch.cat([self.timestamp, ts])
        self.hits = torch.cat([self.hits, torch.zeros(n_new, device=self.device, dtype=torch.long)])
        self._t += 1
        self._evict_if_needed()
        return True

    @torch.no_grad()
    def _evict_if_needed(self) -> None:
        """Evict least-useful entries: low (hits) and old (timestamp)."""
        n = self.K.shape[0]
        if n <= self.max_entries:
            return
        # usefulness score: prefer frequently-hit and recent entries.
        recency = self.timestamp.float() / max(1, self._t)
        usefulness = self.hits.float() + recency  # simple, effective
        keep = torch.topk(usefulness, self.max_entries, largest=True).indices
        keep, _ = torch.sort(keep)
        self.K = self.K[keep]
        self.V = self.V[keep]
        self.timestamp = self.timestamp[keep]
        self.hits = self.hits[keep]

    # -- read (non-differentiable retrieval) ------------------------------
    @torch.no_grad()
    def read(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve top-k memory K/V for the given queries.

        q: (n_q, n_kv_heads, head_dim). Returns retrieved (K_ret, V_ret) of shape
        (n_q, topk, n_kv_heads, head_dim), to be concatenated into an attention
        layer's local K/V. Empty store → empty tensors (caller falls back to
        local attention only).
        """
        n_mem = self.K.shape[0]
        if n_mem == 0:
            empty = torch.zeros(q.shape[0], 0, self.n_kv_heads, self.head_dim,
                                device=q.device, dtype=q.dtype)
            return empty, empty
        k = min(self.topk, n_mem)
        # cosine similarity per head, averaged over heads → (n_q, n_mem)
        qn = F.normalize(q.to(self.dtype), dim=-1)            # (n_q, H, D)
        Kn = F.normalize(self.K, dim=-1)                       # (n_mem, H, D)
        sim = torch.einsum("qhd,mhd->qhm", qn, Kn).mean(dim=1) # (n_q, n_mem)
        top = torch.topk(sim, k, dim=-1).indices               # (n_q, k)
        # update hit counts for retrieved entries
        self.hits.index_add_(0, top.reshape(-1),
                             torch.ones(top.numel(), device=self.device, dtype=torch.long))
        K_ret = self.K[top]   # (n_q, k, H, D)
        V_ret = self.V[top]
        return K_ret.to(q.dtype), V_ret.to(q.dtype)


def _benchmark() -> None:
    """Prove there is no latency-vs-memory tradeoff at personal scale."""
    import time

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    n_kv_heads, head_dim, topk = 8, 128, 8
    print(f"VanitasKVMemory benchmark on '{device}'  (heads={n_kv_heads}, dim={head_dim}, topk={topk})")
    print(f"{'entries':>8} | {'store size':>10} | {'1-query recall':>16}")
    print("-" * 44)

    for n in (100, 1_000, 10_000):
        mem = VanitasKVMemory(max_entries=n, n_kv_heads=n_kv_heads, head_dim=head_dim,
                              topk=topk, device=device)
        # fill it
        k = torch.randn(n, n_kv_heads, head_dim)
        v = torch.randn(n, n_kv_heads, head_dim)
        mem.write(k, v, salience=1.0)

        q = torch.randn(1, n_kv_heads, head_dim, device=device)
        # warmup
        for _ in range(5):
            mem.read(q)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        iters = 100
        for _ in range(iters):
            mem.read(q)
        if device == "cuda":
            torch.cuda.synchronize()
        per_query_ms = (time.time() - t0) / iters * 1000
        mb = mem.memory_bytes() / 1e6
        print(f"{n:>8,} | {mb:>8.2f}MB | {per_query_ms:>13.3f}ms")

    print("\nTakeaway: at one-user scale (100s-1000s of memories), recall is sub-ms "
          "and the store is a few MB. No latency-vs-memory tradeoff at this scale.")


if __name__ == "__main__":
    _benchmark()
