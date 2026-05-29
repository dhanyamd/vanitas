"""Upload trained Vanitas checkpoints from Modal volume to Hugging Face Hub.

Requires a Modal secret named `huggingface` containing `HF_TOKEN`.
Optionally set `HF_REPO_ID`, defaulting to `md13/vanitas-sft`.
"""
import os
import json
import modal

app = modal.App("vanitas-upload")

checkpoints_volume = modal.Volume.from_name("vanitas-checkpoints")

image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub")


@app.function(
    image=image,
    volumes={"/root/checkpoints": checkpoints_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=600,
)
def upload_to_hf():
    from huggingface_hub import HfApi, login

    hf_token = os.environ.get("HF_TOKEN", "")
    login(token=hf_token)

    api = HfApi()
    repo_id = os.environ.get("HF_REPO_ID", "md13/vanitas-sft")

    # List what's on the volume
    print("📂 Checkpoint files on volume:")
    for f in os.listdir("/root/checkpoints"):
        size_mb = os.path.getsize(f"/root/checkpoints/{f}") / (1024 * 1024)
        print(f"   {f} ({size_mb:.1f} MB)")

    # Create the repo
    api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
    print(f"\n✅ Repo {repo_id} ready")

    # Upload best model
    best_path = "/root/checkpoints/best_model.pt"
    if os.path.exists(best_path):
        print("⬆️  Uploading best_model.pt...")
        api.upload_file(
            path_or_fileobj=best_path,
            path_in_repo="best_model.pt",
            repo_id=repo_id,
        )
        print("  ✅ Uploaded best_model.pt")

    # Upload final model
    final_path = "/root/checkpoints/final_model.pt"
    if os.path.exists(final_path):
        print("⬆️  Uploading final_model.pt...")
        api.upload_file(
            path_or_fileobj=final_path,
            path_in_repo="final_model.pt",
            repo_id=repo_id,
        )
        print("  ✅ Uploaded final_model.pt")

    # Upload config as JSON (static, matching cloud training config)
    config_dict = {
        "mel_bins": 80,
        "sample_rate": 16000,
        "hop_length": 160,
        "perception_dim": 512,
        "perception_layers": 12,
        "perception_state_dim": 64,
        "perception_conv_dim": 4,
        "perception_expand": 2,
        "gate_hidden_dim": 256,
        "cognition_dim": 512,
        "cognition_heads": 8,
        "cognition_layers": 8,
        "cognition_vocab_dim": 512,
        "memory_dim": 512,
        "production_dim": 512,
        "production_layers": 12,
        "production_state_dim": 64,
        "dropout": 0.1,
        "adapter_rank": 8,
    }
    config_json = json.dumps(config_dict, indent=2)
    api.upload_file(
        path_or_fileobj=config_json.encode(),
        path_in_repo="config.json",
        repo_id=repo_id,
    )
    print("  ✅ Uploaded config.json")

    # Model card
    model_card = """---
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

Supervised fine-tuned model for real-time spoken dialogue, trained on [kyutai/DailyTalkContiguous](https://huggingface.co/datasets/kyutai/DailyTalkContiguous).

## Architecture
- **Perception Stream**: Mamba-2 SSM (12 layers, d=512)
- **Cognition Core**: Sparse Attention (8 layers, d=512)
- **Production Stream**: Mamba-2 + Flow Matching (12 layers, d=512)

## Training
- **Dataset**: kyutai/DailyTalkContiguous (2,286 dialogues)
- **Epochs**: 50
- **Batch Size**: 16
- **Learning Rate**: 2e-4
- **Hardware**: NVIDIA A100 (Modal Cloud)

## Files
- `best_model.pt` — Checkpoint with the lowest validation loss
- `final_model.pt` — Checkpoint after completing all 50 epochs
- `config.json` — Model configuration
"""
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
    )
    print("  ✅ Uploaded README.md")

    print(f"\n🎉 Model published at: https://huggingface.co/{repo_id}")


@app.local_entrypoint()
def main():
    print("🚀 Uploading checkpoints to Hugging Face...")
    upload_to_hf.remote()
    print("✅ Done!")
