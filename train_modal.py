"""
Vanitas Cloud Training — Modal Deployment Script
=================================================
Usage:
  modal run train_modal.py             # Spawn A100 training on real data
  modal run --detach train_modal.py    # Close laptop safely after launch
"""
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vanitas.train")

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import modal

# ---------------------------------------------------------------------------
# Modal Image — PyTorch base with CUDA toolkit
# ---------------------------------------------------------------------------
vanitas_image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel")
    .apt_install("libsndfile1", "git")
    .run_commands(
        "pip install -q uv && "
        "uv pip install --system "
        "'numpy>=1.26.0' "
        "'scipy>=1.12.0' "
        "'librosa>=0.10.0' "
        "'torchaudio>=2.4.0' "
        "'einops>=0.8.0' "
        "'datasets>=3.0.0' "
        "'huggingface_hub>=0.20.0' "
        "'soundfile>=0.12.1' "
        "'safetensors>=0.4.0'"
    )
    # Copy the vanitas package into the container
    .add_local_dir(
        str(Path(__file__).resolve().parent / "vanitas"),
        remote_path="/root/vanitas",
    )
)

app = modal.App(name="vanitas-training")

# Persistent volume for checkpoints — survives across runs
checkpoints_volume = modal.Volume.from_name("vanitas-checkpoints", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("vanitas-hf-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# Cloud Training Function (runs on A100)
# ---------------------------------------------------------------------------
@app.function(
    image=vanitas_image,
    gpu="A100",
    timeout=21600,  # 6 hours; avoids timeout near the end of 50-epoch runs
    volumes={
        "/root/checkpoints": checkpoints_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
)
def train_cloud(epochs: int = 50, batch_size: int = 16, lr: float = 2e-4):
    """Runs the full training loop on a dedicated A100 GPU in the cloud."""
    import torch
    from torch.optim.lr_scheduler import CosineAnnealingLR

    print("=" * 70)
    print("⚡ CLOUD GPU TRAINING INITIATED (MODAL A100) ⚡")
    print("=" * 70)

    # Imports inside the container (vanitas is at /root/vanitas)
    sys.path.insert(0, "/root")
    from vanitas.config import GlobalConfig
    from vanitas.model.vanitas import VanitasModel
    from vanitas.model.config import VanitasModelConfig
    from vanitas.training.dataset import SpokenDialogueDataset
    from vanitas.training.trainer import SpokenDialogueTrainer

    # ── Step 1: CUDA verification ──────────────────────────────────────
    print(f"\n[1/5] CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"      GPU: {torch.cuda.get_device_name(0)}")
        print(f"      VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Step 2: Model init ─────────────────────────────────────────────
    print("\n[2/5] Initializing Vanitas Model (~85M params)...")
    model_config = VanitasModelConfig(
        perception_dim=512,
        perception_layers=12,
        perception_state_dim=64,
        gate_hidden_dim=256,
    )
    model = VanitasModel(model_config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Parameters: {total_params:,} (~{total_params / 1e6:.1f}M)")

    # ── Step 3: Dataset (REAL DATA — NO MOCK) ──────────────────────────
    print("\n[3/5] Loading REAL DailyTalkContiguous audio from Hugging Face...")
    print("      Hugging Face cache is persisted in the vanitas-hf-cache Modal volume.")
    config = GlobalConfig()
    config.checkpoints_dir = Path("/root/checkpoints")
    config.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SpokenDialogueDataset(config=config, model_config=model_config, split="train", use_mock=False)
    val_ds = SpokenDialogueDataset(config=config, model_config=model_config, split="val", use_mock=False)
    print(f"      Train: {len(train_ds)} dialogues | Val: {len(val_ds)} dialogues")
    
    if len(train_ds) == 0:
        raise RuntimeError("FATAL: No real training data loaded! Refusing to train on empty dataset.")

    # ── Step 4: Train ──────────────────────────────────────────────────
    print(f"\n[4/5] Starting training: {epochs} epochs, batch={batch_size}, lr={lr}")
    trainer = SpokenDialogueTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        use_wandb=False,
        project_name="vanitas-perception-stream",
    )

    start_epoch = 0
    best_path = "/root/checkpoints/best_model.pt"
    if os.path.exists(best_path):
        try:
            print(f"      Found existing checkpoint at {best_path}. Resuming training...")
            checkpoint = torch.load(best_path, map_location=trainer.device, weights_only=False)
            
            # Load model state dict. New production/vocoder layers can be
            # initialized while preserving the already trained streams.
            missing, unexpected = trainer.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if missing:
                print(f"      New parameters initialized from scratch: {len(missing)}")
            if unexpected:
                print(f"      Ignored checkpoint parameters no longer used: {len(unexpected)}")

            outdated_production = any(k.startswith("flow_head.mel_mlp") for k in missing) or any(
                k.startswith("vocoder.dummy_conv") for k in unexpected
            )
            if outdated_production:
                print("      Outdated production checkpoint detected; resetting production/vocoder/speak-gate modules.")
                modules_to_reset = [
                    trainer.model.production_stream,
                    trainer.model.flow_head,
                    trainer.model.vocoder,
                    trainer.model.gates.gate_cg,
                ]
                for module in modules_to_reset:
                    for child in module.modules():
                        if hasattr(child, "reset_parameters"):
                            child.reset_parameters()
                trainer.model.flow_head.mel_mlp[-1].weight.data.zero_()
                trainer.model.flow_head.mel_mlp[-1].bias.data.zero_()
                trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=lr, weight_decay=1e-4)
                trainer.scheduler = CosineAnnealingLR(trainer.optimizer, T_max=epochs, eta_min=1e-6)
                start_epoch = 0
            elif "optimizer_state_dict" in checkpoint:
                try:
                    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                except Exception as opt_e:
                    print(f"      Optimizer state incompatible with new layers; restarting optimizer: {opt_e}")
                
                start_epoch = checkpoint.get("epoch", -1) + 1
                print(f"      Successfully loaded checkpoint from epoch {start_epoch - 1}. Fast-forwarding LR scheduler...")
                
                # Fast-forward LR scheduler to the correct epoch
                for _ in range(start_epoch):
                    trainer.scheduler.step()
        except Exception as e:
            print(f"      ⚠️ Failed to load checkpoint: {e}. Starting from scratch.")
            start_epoch = 0

    trainer.fit(start_epoch=start_epoch)

    # Persist checkpoints and dataset cache to Modal Volumes
    checkpoints_volume.commit()
    hf_cache_volume.commit()
    print("\n✅ Training complete! Checkpoints saved to vanitas-checkpoints volume.")

    # ── Step 5: Push model to Hugging Face ─────────────────────────────
    print("\n[5/5] Pushing trained model to Hugging Face Hub...")
    try:
        from huggingface_hub import HfApi, login
        
        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            print("  ⚠️ HF_TOKEN is not set; skipping Hugging Face upload.")
            print("     Checkpoints remain saved in the Modal volume.")
            print("=" * 70)
            return

        login(token=hf_token)
        
        api = HfApi()
        repo_id = os.environ.get("HF_REPO_ID", "md13/vanitas-sft")
        
        # Create the repo (or use existing)
        api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
        
        # Upload best model checkpoint
        best_path = "/root/checkpoints/best_model.pt"
        final_path = "/root/checkpoints/final_model.pt"
        
        if os.path.exists(best_path):
            api.upload_file(
                path_or_fileobj=best_path,
                path_in_repo="best_model.pt",
                repo_id=repo_id,
            )
            print(f"  ✅ Uploaded best_model.pt to {repo_id}")
            
        if os.path.exists(final_path):
            api.upload_file(
                path_or_fileobj=final_path,
                path_in_repo="final_model.pt",
                repo_id=repo_id,
            )
            print(f"  ✅ Uploaded final_model.pt to {repo_id}")
        
        # Upload model config as JSON
        import json
        from dataclasses import asdict
        config_dict = asdict(model_config)
        config_json = json.dumps(config_dict, indent=2)
        api.upload_file(
            path_or_fileobj=config_json.encode(),
            path_in_repo="config.json",
            repo_id=repo_id,
        )
        print(f"  ✅ Uploaded config.json to {repo_id}")
        
        # Create a model card
        model_card = f"""---
tags:
- vanitas
- spoken-dialogue
- mamba-ssm
- flow-matching
license: mit
datasets:
- kyutai/DailyTalkContiguous
---

# Vanitas SFT Model

Supervised fine-tuned model for real-time spoken dialogue, trained on the [kyutai/DailyTalkContiguous](https://huggingface.co/datasets/kyutai/DailyTalkContiguous) dataset.

## Architecture
- **Perception Stream**: Mamba-2 SSM ({model_config.perception_layers} layers, d={model_config.perception_dim})
- **Cognition Core**: Sparse Attention ({model_config.cognition_layers} layers, d={model_config.cognition_dim})
- **Production Stream**: Mamba-2 + Flow Matching ({model_config.production_layers} layers, d={model_config.production_dim})
- **Parameters**: {total_params:,} (~{total_params / 1e6:.1f}M)

## Training
- **Dataset**: kyutai/DailyTalkContiguous ({len(train_ds)} train / {len(val_ds)} val dialogues)
- **Epochs**: {epochs}
- **Batch Size**: {batch_size}
- **Learning Rate**: {lr}
- **Hardware**: NVIDIA A100 (Modal Cloud)
"""
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
        )
        print(f"  ✅ Uploaded README.md to {repo_id}")
        print(f"\n🎉 Model published at: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"  ⚠️ Failed to push to HF: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 70)


# ---------------------------------------------------------------------------
# Entrypoint — `modal run train_modal.py` triggers this
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(epochs: int = 50, batch_size: int = 16, lr: float = 2e-4):
    """Launch cloud training. Use `modal run --detach train_modal.py` to close your laptop."""
    print("🚀 Launching A100 training job on Modal...")
    print("   Training on REAL kyutai/DailyTalkContiguous dataset (NO MOCK DATA)")
    print("   Checkpoints will be saved to the Modal volume.")
    print("   Run upload_to_hf.py after training to publish the corrected checkpoint.")
    print("   You can safely close your terminal after the image builds.")
    print("   Monitor the run from your Modal dashboard.\n")
    # Using .spawn() instead of .remote() ensures the detached run is not liable
    # to cancellation when the local process exits/disconnects.
    train_cloud.spawn(epochs=epochs, batch_size=batch_size, lr=lr)
    print("\n✅ Training job spawned successfully on the cloud!")
