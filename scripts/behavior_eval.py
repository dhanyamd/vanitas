"""Behavior-level evaluation for Vanitas turn-taking.

Measures gate behavior on a small dataset slice: think-gate boundary detection,
speak-gate agent-activity detection, overlap ratio, and simple latency proxies.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vanitas.config import GlobalConfig
from vanitas.model.config import VanitasModelConfig
from vanitas.model.vanitas import VanitasModel
from vanitas.training.dataset import SpokenDialogueDataset, pad_collate_fn


def _config_from_checkpoint(checkpoint: dict) -> VanitasModelConfig:
    raw_config = checkpoint.get("config") or checkpoint.get("model_config")
    if isinstance(raw_config, VanitasModelConfig):
        return raw_config
    if isinstance(raw_config, dict):
        valid_keys = VanitasModelConfig.__dataclass_fields__.keys()
        return VanitasModelConfig(**{k: v for k, v in raw_config.items() if k in valid_keys})
    return VanitasModelConfig()


def _binary_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float) -> dict[str, float]:
    pred_bin = pred >= threshold
    target_bin = target >= 0.5
    tp = (pred_bin & target_bin).sum().item()
    fp = (pred_bin & ~target_bin).sum().item()
    fn = (~pred_bin & target_bin).sum().item()
    tn = (~pred_bin & ~target_bin).sum().item()
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    acc = (tp + tn) / max(1, tp + fp + fn + tn)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Vanitas conversational behavior gates.")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mock", action="store_true", help="Use synthetic mock data.")
    parser.add_argument("--threshold", type=float, default=0.45)
    args = parser.parse_args()

    cfg = GlobalConfig()
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_cfg = _config_from_checkpoint(checkpoint)
        model = VanitasModel(model_cfg)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    else:
        model_cfg = VanitasModelConfig()
        model = VanitasModel(model_cfg)
        print(f"WARNING: checkpoint not found at {ckpt_path}; evaluating random initialized model.")

    model.eval()
    dataset = SpokenDialogueDataset(
        config=cfg,
        model_config=model_cfg,
        split=args.split,
        max_samples=args.samples,
        use_mock=args.mock,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_fn)

    think_preds = []
    think_targets = []
    speak_preds = []
    speak_targets = []
    overlaps = []

    with torch.no_grad():
        for batch in loader:
            agent_shifted = torch.full_like(batch["agent_mel"], -5.0)
            if agent_shifted.size(1) > 1:
                agent_shifted[:, 1:, :] = batch["agent_mel"][:, :-1, :]
            outputs = model(batch["masked_mel"], agent_mel_frames=agent_shifted)
            lengths = batch["lengths"]
            for i, length in enumerate(lengths.tolist()):
                think_preds.append(outputs["think_gate"][i, :length].cpu())
                speak_preds.append(outputs["speak_gate"][i, :length].cpu())
                think_targets.append(batch["turn_target"][i, :length].cpu())
                speak_targets.append(batch["agent_active"][i, :length].cpu())
                # prosody feature index 6 is overlap ratio.
                overlaps.append(float(batch["prosody_features"][i, 6].item()))

    think_pred = torch.cat(think_preds, dim=0)
    think_target = torch.cat(think_targets, dim=0)
    speak_pred = torch.cat(speak_preds, dim=0)
    speak_target = torch.cat(speak_targets, dim=0)

    think_metrics = _binary_metrics(think_pred, think_target, args.threshold)
    speak_metrics = _binary_metrics(speak_pred, speak_target, args.threshold)
    think_bce = F.binary_cross_entropy(think_pred.clamp(1e-6, 1 - 1e-6), think_target).item()
    speak_bce = F.binary_cross_entropy(speak_pred.clamp(1e-6, 1 - 1e-6), speak_target).item()

    print("\nVanitas Behavior Evaluation")
    print(f"- samples: {len(dataset)}")
    print(f"- threshold: {args.threshold:.2f}")
    print(f"- mean overlap ratio: {sum(overlaps) / max(1, len(overlaps)):.4f}")
    print(f"- think target positive rate: {think_target.mean().item():.4f}")
    print(f"- speak target positive rate: {speak_target.mean().item():.4f}")
    print(f"- think pred mean: {think_pred.mean().item():.4f}, BCE: {think_bce:.4f}")
    print(f"- speak pred mean: {speak_pred.mean().item():.4f}, BCE: {speak_bce:.4f}")
    print(f"- think metrics: {think_metrics}")
    print(f"- speak metrics: {speak_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
