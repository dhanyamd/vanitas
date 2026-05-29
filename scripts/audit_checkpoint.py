"""Audit a Vanitas checkpoint for inference/training compatibility.

This script is intentionally lightweight: it does not run generation. It checks
whether a checkpoint was trained before the corrected flow-matching/vocoder path
was added, and reports which model should be retrained before clear speech can
be expected.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vanitas.model.config import VanitasModelConfig
from vanitas.model.vanitas import VanitasModel


def _config_from_checkpoint(checkpoint: dict) -> VanitasModelConfig:
    raw_config = checkpoint.get("config") or checkpoint.get("model_config")
    if isinstance(raw_config, VanitasModelConfig):
        return raw_config
    if isinstance(raw_config, dict):
        valid_keys = VanitasModelConfig.__dataclass_fields__.keys()
        return VanitasModelConfig(**{k: v for k, v in raw_config.items() if k in valid_keys})
    return VanitasModelConfig()


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a Vanitas checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pt",
        help="Path to checkpoint .pt file.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: checkpoint not found: {checkpoint_path}")
        return 2

    print(f"Loading checkpoint: {checkpoint_path}")
    # Local project checkpoints include a VanitasModelConfig dataclass. We load
    # full metadata here because this script is for trusted checkpoints produced
    # by this repo, not arbitrary downloaded files.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        print("ERROR: checkpoint does not contain a valid state dict.")
        return 2

    config = _config_from_checkpoint(checkpoint if isinstance(checkpoint, dict) else {})
    model = VanitasModel(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    total_state_params = sum(v.numel() for v in state_dict.values() if torch.is_tensor(v))
    old_dummy_vocoder = [k for k in state_dict if k.startswith("vocoder.dummy_conv")]
    has_corrected_flow = any(k.startswith("flow_head.mel_mlp") for k in state_dict)
    missing_corrected_flow = [k for k in missing if k.startswith("flow_head.mel_mlp")]
    missing_vocoder = [k for k in missing if k.startswith("vocoder.")]

    print("\nCheckpoint Summary")
    print(f"- tensor parameters in checkpoint: {total_state_params:,}")
    print(f"- saved epoch: {checkpoint.get('epoch', 'unknown') if isinstance(checkpoint, dict) else 'unknown'}")
    print(f"- saved validation loss: {checkpoint.get('val_loss', 'unknown') if isinstance(checkpoint, dict) else 'unknown'}")
    print(f"- missing keys when loaded into current code: {len(missing)}")
    print(f"- unexpected keys from checkpoint: {len(unexpected)}")

    print("\nProduction Path Audit")
    print(f"- corrected flow conditioning keys present: {has_corrected_flow}")
    print(f"- old dummy vocoder keys present: {bool(old_dummy_vocoder)}")
    print(f"- missing corrected flow keys: {len(missing_corrected_flow)}")
    print(f"- missing vocoder keys: {len(missing_vocoder)}")

    if old_dummy_vocoder or missing_corrected_flow:
        print("\nVERDICT: OUTDATED_PRODUCTION_CHECKPOINT")
        print("This checkpoint can preserve useful perception/cognition weights, but")
        print("it cannot be expected to produce clear speech without corrective")
        print("training of the production stream, flow head, vocoder path, and speak gate.")
        return 1

    if missing or unexpected:
        print("\nVERDICT: LOADS_WITH_COMPATIBILITY_WARNINGS")
        print("The checkpoint matches the corrected production path but still has")
        print("non-fatal key differences. Run a generation diagnostic before deployment.")
        return 0

    print("\nVERDICT: CURRENT_ARCHITECTURE_COMPATIBLE")
    print("The checkpoint matches the corrected model architecture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
