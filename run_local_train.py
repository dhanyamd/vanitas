import torch
from pathlib import Path

# Project imports
from vanitas.config import GlobalConfig
from vanitas.model.vanitas import VanitasModel
from vanitas.model.config import VanitasModelConfig
from vanitas.training.dataset import SpokenDialogueDataset
from vanitas.training.trainer import SpokenDialogueTrainer

def main(epochs: int = 5, batch_size: int = 8, lr: float = 2e-4):
    # Initialize config and model
    config = GlobalConfig()
    config.checkpoints_dir = Path("./checkpoints")
    config.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = VanitasModelConfig(
        perception_dim=512,
        perception_layers=12,
        perception_state_dim=64,
        gate_hidden_dim=256,
    )
    model = VanitasModel(model_cfg)

    # Datasets (real, no mock)
    train_ds = SpokenDialogueDataset(config=config, model_config=model_cfg, split="train", use_mock=False)
    val_ds = SpokenDialogueDataset(config=config, model_config=model_cfg, split="val", use_mock=False)

    trainer = SpokenDialogueTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        use_wandb=False,
    )
    trainer.fit()

if __name__ == "__main__":
    # Quick run with a few epochs for demo purposes
    main(epochs=3, batch_size=4)
