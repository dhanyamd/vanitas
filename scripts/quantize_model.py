#!/usr/bin/env python
"""Quantize a Vanitas checkpoint.

Usage:
    uv run python scripts/quantize_model.py \
        --ckpt checkpoints/best_model.pt \
        --out checkpoints/best_model_int8.pt \
        [--mode ptq|qat|weight]

* ``ptq`` – static post‑training quantization (default).
* ``qat`` – runs a single‑epoch QAT fine‑tune (requires a training loader).
* ``weight`` – weight‑only 4‑bit quantization (activations stay FP16).

The script writes the quantized ``state_dict`` together with the original config
so that the inference server can load it transparently.
"""

import argparse
import pathlib
import sys

import torch

# Ensure project root is on PYTHONPATH (when run from scripts folder)
project_root = pathlib.Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from vanitas.model.vanitas import VanitasModel
from vanitas.model.config import VanitasModelConfig
from vanitas.quantization.quantizer import (
    quantize_static,
    quantize_qat,
    quantize_weight_only_4bit,
)
from vanitas.quantization.calib_loader import get_calib_loader
from vanitas.training.dataset import VanitasDataset  # existing dataset class for QAT


def load_checkpoint(path: pathlib.Path):
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    config = ckpt["config"] if "config" in ckpt else None
    model = VanitasModel(config) if config else None
    if model is not None:
        model.load_state_dict(ckpt["model_state_dict"])
    return model, config, ckpt


def save_quantized(model: torch.nn.Module, config, original_ckpt, out_path: pathlib.Path):
    # Preserve the same top‑level dict structure the server expects.
    quant_ckpt = {
        "model_state_dict": model.state_dict(),
        "config": config,
    }
    # Keep any extra keys (e.g., optimizer state) if present – optional.
    for k, v in original_ckpt.items():
        if k not in quant_ckpt:
            quant_ckpt[k] = v
    torch.save(quant_ckpt, str(out_path))
    print(f"[quantize] Saved quantized checkpoint to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Quantize Vanitas checkpoint")
    parser.add_argument("--ckpt", required=True, help="Path to original .pt checkpoint")
    parser.add_argument("--out", required=True, help="Path to write quantized checkpoint")
    parser.add_argument(
        "--mode",
        choices=["ptq", "qat", "weight"],
        default="ptq",
        help="Quantization strategy",
    )
    parser.add_argument(
        "--calib-dir",
        default="data/calibration",
        help="Directory with raw wav files for PTQ calibration",
    )
    parser.add_argument(
        "--train-dir",
        default="data/train",
        help="Directory with training data for QAT (VanitasDataset expects wav+txt pairs)",
    )
    args = parser.parse_args()

    ckpt_path = pathlib.Path(args.ckpt)
    out_path = pathlib.Path(args.out)

    model, config, orig_ckpt = load_checkpoint(ckpt_path)
    model.eval()

    if args.mode == "ptq":
        calib_loader = get_calib_loader(args.calib_dir, batch_size=4)
        q_model = quantize_static(model, calib_loader)
    elif args.mode == "qat":
        # Very lightweight fine‑tune – one epoch over a small subset.
        train_dataset = VanitasDataset(root=args.train_dir)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
        q_model = quantize_qat(model, train_loader, epochs=1)
    else:  # weight‑only 4‑bit
        q_model = quantize_weight_only_4bit(model)

    save_quantized(q_model, config, orig_ckpt, out_path)

if __name__ == "__main__":
    main()
