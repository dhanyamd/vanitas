"""Generate a short Vanitas audio diagnostic from a checkpoint.

This is not a product demo. It is a fast sanity check that reports whether a
checkpoint produces non-silent waveform output through the current inference
path and saves a WAV file for listening.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
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


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a short Vanitas diagnostic WAV.")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--out", default="scratch/diagnostic_response.wav")
    parser.add_argument("--frames", type=int, default=160, help="Number of mel frames to synthesize from.")
    parser.add_argument("--fail-on-outdated", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: checkpoint not found: {checkpoint_path}")
        return 2

    # Local project checkpoints include a VanitasModelConfig dataclass. We load
    # full metadata here because this script is for trusted checkpoints produced
    # by this repo, not arbitrary downloaded files.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    old_dummy_vocoder = any(k.startswith("vocoder.dummy_conv") for k in state_dict)
    missing_corrected_flow = not any(k.startswith("flow_head.mel_mlp") for k in state_dict)
    if old_dummy_vocoder or missing_corrected_flow:
        print("WARNING: checkpoint predates the corrected production/vocoder path.")
        print("Generated audio from this checkpoint is expected to be weak/noisy.")
        if args.fail_on_outdated:
            return 1

    config = _config_from_checkpoint(checkpoint if isinstance(checkpoint, dict) else {})
    device = _device()
    model = VanitasModel(config).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Load warning: {len(missing)} missing keys.")
    if unexpected:
        print(f"Load warning: {len(unexpected)} unexpected keys.")
    model.eval()

    # Synthetic user-side log-mel input. This checks the model/vocoder path
    # without requiring microphone access or a dataset download.
    frames = max(16, int(args.frames))
    time_axis = torch.linspace(0.0, 1.0, frames, device=device).view(1, frames, 1)
    mel_bins = torch.linspace(0.0, 1.0, config.mel_bins, device=device).view(1, 1, config.mel_bins)
    mel_input = -4.0 + 1.5 * torch.sin(2.0 * torch.pi * (3.0 * time_axis + mel_bins))
    agent_feedback = torch.full_like(mel_input, -5.0)

    with torch.no_grad():
        outputs = model(mel_input, agent_mel_frames=agent_feedback)
    audio = outputs["audio"]
    if audio is None:
        print("ERROR: model returned no audio.")
        return 2

    audio_np = audio.detach().cpu().float().numpy().reshape(-1)
    audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
    rms = float(np.sqrt(np.mean(audio_np**2))) if audio_np.size else 0.0
    duration = audio_np.size / float(config.sample_rate)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
        sf.write(out_path, audio_np, config.sample_rate)
    except Exception as exc:
        print(f"ERROR: could not write WAV: {exc}")
        return 2

    print("\nAudio Diagnostic")
    print(f"- output: {out_path.resolve()}")
    print(f"- duration: {duration:.2f}s")
    print(f"- rms: {rms:.6f}")
    print(f"- peak: {peak:.6f}")
    print(f"- speak gate mean: {outputs['speak_gate'].detach().cpu().float().mean().item():.4f}")
    print(f"- think gate mean: {outputs['think_gate'].detach().cpu().float().mean().item():.4f}")

    if rms < 0.005 or peak < 0.02:
        print("\nVERDICT: TOO_QUIET_OR_WEAK")
        print("This checkpoint is not ready for a clear live voice demo.")
        return 1

    print("\nVERDICT: NONSILENT_AUDIO_PATH")
    print("The checkpoint produces audible waveform output. Listen to the WAV for quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
