"""Stage 0 — Plumbing sanity check.

Verifies, before any training, that:

  1. SNAC loads and round-trips synthetic audio without crashing.
  2. SNAC's three code resolutions have the expected shape and rate.
  3. (Optional, GPU only) Qwen3-1.7B loads via transformers and generates English.

Run:
    python scripts/v2/stage0_sanity.py
    python scripts/v2/stage0_sanity.py --skip-llm   # SNAC only (works on Mac)
    python scripts/v2/stage0_sanity.py --device cpu # force CPU

Exit code 0 = sanity passed. Non-zero = something is wrong; do NOT proceed to
Stage 1 until this exits 0.
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import torch

from vanitas.v2.codec.snac_wrapper import SNAC_SAMPLE_RATE, decode, encode, load_snac


# ---------------------------------------------------------------------------
# Step 1: SNAC round-trip
# ---------------------------------------------------------------------------

def _make_synthetic_audio(seconds: float, sr: int) -> torch.Tensor:
    """A 220 Hz sine wave + a touch of noise. Enough signal for SNAC to encode."""
    t = torch.arange(int(seconds * sr)) / sr
    wave = 0.6 * torch.sin(2 * math.pi * 220.0 * t) + 0.05 * torch.randn_like(t)
    return wave.unsqueeze(0).unsqueeze(0).float()  # (1, 1, T)


def check_snac(device: str) -> None:
    print(f"\n[SNAC] Loading hubertsiuzdak/snac_24khz on '{device}'...")
    model = load_snac(device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[SNAC] Loaded. params={n_params:,} (~{n_params/1e6:.1f}M)")

    # Encode 2 seconds of audio
    seconds = 2.0
    wav = _make_synthetic_audio(seconds, SNAC_SAMPLE_RATE)
    print(f"[SNAC] Input waveform: shape={tuple(wav.shape)} duration={seconds}s @ {SNAC_SAMPLE_RATE}Hz")

    t0 = time.time()
    enc = encode(model, wav)
    t_enc = time.time() - t0
    print(f"[SNAC] Encode took {t_enc*1000:.1f} ms")
    for i, c in enumerate(enc.codes):
        rate = c.shape[-1] / seconds
        print(f"[SNAC]   level {i}: shape={tuple(c.shape)}  approx_rate={rate:.1f} Hz")

    # Sanity: SNAC's three levels should be at 12 / 24 / 48 Hz (verified).
    expected_rates = [12, 24, 48]
    for i, (c, expected) in enumerate(zip(enc.codes, expected_rates)):
        actual = c.shape[-1] / seconds
        if abs(actual - expected) > expected * 0.25:
            raise RuntimeError(
                f"SNAC level {i} unexpected rate: got {actual:.1f} Hz, expected ~{expected} Hz"
            )

    t0 = time.time()
    recon = decode(model, enc)
    t_dec = time.time() - t0
    print(f"[SNAC] Decode took {t_dec*1000:.1f} ms")
    print(f"[SNAC] Reconstructed waveform: shape={tuple(recon.shape)}")

    # Reconstruction RMS sanity: the decoded waveform should have non-trivial amplitude.
    rms = torch.sqrt((recon.float() ** 2).mean()).item()
    if rms < 0.01:
        raise RuntimeError(f"Reconstruction RMS is suspiciously low: {rms:.4f}")
    print(f"[SNAC] Reconstruction RMS = {rms:.4f}  ✓")


# ---------------------------------------------------------------------------
# Step 2: Qwen3-1.7B load + tiny generation (GPU only)
# ---------------------------------------------------------------------------

def check_qwen3(device: str) -> None:
    if device == "cpu":
        print("\n[Qwen3] Skipping LM check on CPU (it would be too slow to be informative).")
        return
    print(f"\n[Qwen3] Loading Qwen/Qwen3-1.7B on '{device}'...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers not installed. `pip install transformers`") from exc

    dtype = torch.float16 if device != "cpu" else torch.float32
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B", torch_dtype=dtype
    ).to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Qwen3] Loaded. params={n_params:,} (~{n_params/1e9:.2f}B)")

    prompt = "Hello. Briefly describe what a neural audio codec does."
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"[Qwen3] Sample generation:\n  {text!r}")
    if len(text.strip()) < 5:
        raise RuntimeError("Qwen3 generated empty/trivial output. Something is wrong.")
    print("[Qwen3] Generation looks coherent ✓")


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Vanitas-SLM Stage 0 sanity check.")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Compute device (default: auto-detect)")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip Qwen3 load (useful on Mac for codec-only check)")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print("=" * 70)
    print("Vanitas-SLM Stage 0 — Sanity Check")
    print(f"Device: {device}")
    print("=" * 70)

    try:
        check_snac(device)
        if not args.skip_llm:
            check_qwen3(device)
    except Exception as exc:
        print(f"\n❌ Stage 0 FAILED: {exc!r}")
        return 1

    print("\n" + "=" * 70)
    print("✅ Stage 0 passed. Plumbing is sound; safe to proceed to Stage 1.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
