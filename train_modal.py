import os
import sys
import click
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import Modal
try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

# Modal App definition (only defined if modal is installed)
if MODAL_AVAILABLE:
    # Standard image definition with GPU libraries
    vanitas_image = (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("libsndfile1", "git") # libsndfile is required by soundfile/torchaudio
        .pip_install(
            "numpy>=1.26.0",
            "scipy>=1.12.0",
            "librosa>=0.10.0",
            "torch>=2.2.0",
            "torchaudio>=2.2.0",
            "einops>=0.8.0",
            "datasets>=3.0.0",
            "huggingface_hub>=0.20.0",
            "wandb>=0.18.0",
            "click>=8.1.0",
            "soundfile>=0.12.1",
            "chromadb>=0.4.0"
        )
        # Install optimized Mamba CUDA kernels - these compile cleanly inside CUDA environment
        .pip_install("causal-conv1d>=1.4.0", "mamba-ssm>=2.2.0", pre=True)
        .add_local_dir(
            str(Path(__file__).resolve().parent / "vanitas"),
            remote_path="/root/vanitas"
        )
    )
    
    app = modal.App(name="vanitas-training")
    
    # Define a persistent volume for storing model checkpoints securely in the cloud
    checkpoints_volume = modal.Volume.from_name("vanitas-checkpoints", create_if_missing=True)


# Local execution helper
def run_local_training(epochs: int, batch_size: int):
    """Executes a short, local training dry-run on Apple Silicon or CPU using mock conversations."""
    print("="*70)
    print("🔬 RUNNING LOCAL DRY-RUN PROTOTYPE TRAINING 🔬")
    print("="*70)
    
    from vanitas.config import GlobalConfig
    from vanitas.model.vanitas import VanitasModel
    from vanitas.training.dataset import SpokenDialogueDataset
    from vanitas.training.trainer import SpokenDialogueTrainer
    
    config = GlobalConfig()
    
    print("\n[Step 1] Initializing Vanitas Model Configuration...")
    from vanitas.model.config import VanitasModelConfig
    model_config = VanitasModelConfig(
        perception_layers=2, # Scale layers down to 2 for instant compilation and execution
        cognition_layers=2,
        production_layers=2,
        perception_dim=128,   # Scale dim to 128
        cognition_dim=128,
        production_dim=128,
        memory_dim=128,
        gate_hidden_dim=64
    )
    model = VanitasModel(model_config)
    print("Vanitas model structure successfully initialized.")
    
    print("\n[Step 2] Loading local mock training and validation datasets...")
    # Force use of mock synthetic data to guarantee it runs offline instantly
    train_ds = SpokenDialogueDataset(config=config, model_config=model_config, split="train", max_samples=4, use_mock=True)
    val_ds = SpokenDialogueDataset(config=config, model_config=model_config, split="val", max_samples=2, use_mock=True)
    print(f"Loaded {len(train_ds)} train and {len(val_ds)} validation mock dialogues.")
    
    print(f"\n[Step 3] Initializing SpokenDialogueTrainer for {epochs} epochs...")
    trainer = SpokenDialogueTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
        lr=5e-4,
        batch_size=batch_size,
        epochs=epochs,
        use_wandb=False # No wandb for local debug
    )
    
    print("\n[Step 4] Starting training loop...")
    trainer.fit()
    
    print("\n🟢 Local diagnostic training dry-run finished successfully!")
    print(f"Checkpoints saved in: '{config.checkpoints_dir}'")
    print("="*70)


# Modal Cloud Entry Point
if MODAL_AVAILABLE:
    @app.function(
        image=vanitas_image,
        gpu="A100", # Secure NVIDIA A100 GPU (40GB or 80GB)
        timeout=7200, # 2 hours max run time
        volumes={"/root/checkpoints": checkpoints_volume}
    )
    def train_cloud(epochs: int = 50, batch_size: int = 16, lr: float = 2e-4, use_wandb: bool = False):
        """Runs the full academic training loop in the cloud on a dedicated A100 GPU."""
        print("="*70)
        print("⚡ CLOUD GPU TRAINING INITIATED (MODAL A100) ⚡")
        print("="*70)
        
        import torch
        from vanitas.config import GlobalConfig
        from vanitas.model.vanitas import VanitasModel
        from vanitas.model.config import VanitasModelConfig
        from vanitas.training.dataset import SpokenDialogueDataset
        from vanitas.training.trainer import SpokenDialogueTrainer
        
        # Override config paths for container directory structures
        config = GlobalConfig()
        config.checkpoints_dir = Path("/root/checkpoints")
        config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n[Step 1] Initializing CUDA Device Verification...")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            
        print("\n[Step 2] Initializing full academic Vanitas Model (~85M params)...")
        # Full hyperparameter targets for paper benchmarks
        model_config = VanitasModelConfig(
            perception_dim=512,
            perception_layers=12,
            perception_state_dim=64,
            gate_hidden_dim=256
        )
        model = VanitasModel(model_config)
        
        print("\n[Step 3] Downloading 'kyutai/DailyTalkContiguous' dataset from Hugging Face...")
        train_ds = SpokenDialogueDataset(config=config, model_config=model_config, split="train", use_mock=False)
        val_ds = SpokenDialogueDataset(config=config, model_config=model_config, split="val", use_mock=False)
        print(f"Downloaded {len(train_ds)} train dialogues and {len(val_ds)} validation dialogues.")
        
        # Log parameter count
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total Model Parameter Count: {total_params:,} (~{total_params/1e6:.1f}M)")
        
        print("\n[Step 4] Starting Cloud Trainer...")
        trainer = SpokenDialogueTrainer(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            config=config,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            use_wandb=use_wandb,
            project_name="vanitas-perception-stream"
        )
        
        trainer.fit()
        
        # Commit checkpoint volume changes to cloud storage
        checkpoints_volume.commit()
        print("\n⚡ Cloud GPU training finalized! Checkpoints securely synced to persistent volume. ⚡")
        print("="*70)


# CLI interface to execute script locally or trigger cloud
@click.command()
@click.option("--local", is_flag=True, help="Run a short local training dry-run on Apple Silicon or CPU.")
@click.option("--epochs", default=2, help="Number of training epochs.")
@click.option("--batch-size", default=4, help="Batch size for training.")
@click.option("--lr", default=2e-4, help="Learning rate.")
@click.option("--wandb", is_flag=True, help="Log metrics to Weights & Biases.")
def main(local, epochs, batch_size, lr, wandb):
    """🚀 Modal Runner for training the Vanitas Perception Stream & learned gates."""
    if local:
        run_local_training(epochs=epochs, batch_size=batch_size)
    else:
        if not MODAL_AVAILABLE:
            print("❌ Modal library not found. Install modal via `pip install modal` and setup auth.")
            sys.exit(1)
            
        print("🚀 Deploying A100 GPU training job to Modal cloud...")
        # Triggering cloud function
        with modal.enable_output():
            # Standard cloud launch parameters
            train_cloud.remote(epochs=epochs, batch_size=batch_size, lr=lr, use_wandb=wandb)

if __name__ == "__main__":
    main()
