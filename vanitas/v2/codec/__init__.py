"""SNAC codec wrapper (frozen).

Loads hubertsiuzdak/snac_24khz. Three resolutions at 12/23/47 Hz.
Used as a black-box tokenizer; no training happens here.

Public API:
  - ``load_snac(device="cuda" | "mps" | "cpu")`` returns the model.
  - ``encode(model, waveform)`` returns the 3 SNAC code tensors.
  - ``decode(model, codes)`` returns a waveform.
"""
