import os
import time
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from pathlib import Path

from vanitas.config import GlobalConfig
from vanitas.model.vanitas import VanitasModel
from vanitas.training.dataset import pad_collate_fn
from vanitas.training.losses import JointPerceptionGateLoss

# Set logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vanitas.training.trainer")

# Check if Weights & Biases is installed
WANDB_AVAILABLE = False
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    pass

class SpokenDialogueTrainer:
    """Trainer orchestrator for the Vanitas always-on Perception Stream and learned Gates."""
    
    def __init__(self, 
                 model: VanitasModel,
                 train_dataset,
                 val_dataset,
                 config: GlobalConfig = None,
                 lr: float = 2e-4,
                 batch_size: int = 8,
                 epochs: int = 10,
                 use_wandb: bool = False,
                 project_name: str = "vanitas-architecture"):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config if config is not None else GlobalConfig()
        
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.project_name = project_name
        
        # Determine training device (CUDA -> MPS -> CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        logger.info(f"Trainer engaged. Hardware backend selected: '{self.device}'")
        
        # Loaders with custom pad collate
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=pad_collate_fn,
            num_workers=0 # Keep at 0 to prevent multi-processing overhead locally
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=pad_collate_fn,
            num_workers=0
        )
        
        # Loss function
        self.loss_fn = JointPerceptionGateLoss(
            mel_bins=self.model.config.mel_bins,
            model_dim=self.model.config.perception_dim
        )
        
        # Move model and loss head to device
        self.model.to(self.device)
        self.loss_fn.to(self.device)
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        # Learning Rate Cosine Annealer
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)
        
        # Initialize WandB
        if self.use_wandb:
            wandb.init(
                project=self.project_name,
                config={
                    "learning_rate": self.lr,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                    "device": str(self.device),
                    "perception_layers": self.model.config.perception_layers,
                    "perception_dim": self.model.config.perception_dim,
                }
            )

    def train_epoch(self, epoch: int) -> tuple[float, float, float]:
        """Runs a single epoch of training across all mini-batches."""
        self.model.train()
        total_epoch_loss = 0.0
        total_mel_loss = 0.0
        total_gate_loss = 0.0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch tensors to device
            mel_input = batch["mel_input"].to(self.device)
            masked_mel = batch["masked_mel"].to(self.device)
            mask_indices = batch["mask_indices"].to(self.device)
            turn_target = batch["turn_target"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass: Feed MASKED mel spectrograms into Perception Stream
            # Outputs: perception_outputs (B, T, D), think_gate (B, T, 1)
            perception_outputs, _, think_gate, _, _ = self.model(masked_mel)
            
            # Compute loss
            loss, mel_loss, gate_loss = self.loss_fn(
                perception_outputs=perception_outputs,
                think_gate_preds=think_gate,
                ground_truth_mel=mel_input, # Match reconstruction to unmasked original
                mask_indices=mask_indices,
                turn_targets=turn_target,
                lengths=lengths
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to ensure training stability in SSMs
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Step optimizer
            self.optimizer.step()
            
            # Accumulate logs
            total_epoch_loss += loss.item()
            total_mel_loss += mel_loss.item()
            total_gate_loss += gate_loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} | Batch {batch_idx}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.4f} (Mel: {mel_loss.item():.4f}, Gate: {gate_loss.item():.4f})"
                )
                
        # Average epoch losses
        n_batches = len(self.train_loader)
        avg_loss = total_epoch_loss / n_batches
        avg_mel = total_mel_loss / n_batches
        avg_gate = total_gate_loss / n_batches
        
        epoch_time = time.time() - start_time
        logger.info(
            f"🟢 Epoch {epoch+1} Completed in {epoch_time:.2f}s | "
            f"Avg Loss: {avg_loss:.4f} (Mel: {avg_mel:.4f}, Gate: {avg_gate:.4f})"
        )
        
        return avg_loss, avg_mel, avg_gate

    def evaluate(self) -> tuple[float, float, float]:
        """Evaluates the model over the validation set."""
        self.model.eval()
        total_val_loss = 0.0
        total_mel_loss = 0.0
        total_gate_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                mel_input = batch["mel_input"].to(self.device)
                masked_mel = batch["masked_mel"].to(self.device)
                mask_indices = batch["mask_indices"].to(self.device)
                turn_target = batch["turn_target"].to(self.device)
                lengths = batch["lengths"].to(self.device)
                
                # Forward
                perception_outputs, _, think_gate, _, _ = self.model(masked_mel)
                
                # Loss
                loss, mel_loss, gate_loss = self.loss_fn(
                    perception_outputs=perception_outputs,
                    think_gate_preds=think_gate,
                    ground_truth_mel=mel_input,
                    mask_indices=mask_indices,
                    turn_targets=turn_target,
                    lengths=lengths
                )
                
                total_val_loss += loss.item()
                total_mel_loss += mel_loss.item()
                total_gate_loss += gate_loss.item()
                
        n_batches = len(self.val_loader)
        return total_val_loss / n_batches, total_mel_loss / n_batches, total_gate_loss / n_batches

    def fit(self):
        """Executes the complete training, learning rate schedules, evaluations, and checkpointing loops."""
        logger.info("Initializing SpokenDialogueTrainer fitting routine...")
        best_val_loss = float("inf")
        
        for epoch in range(self.epochs):
            # 1. Train one epoch
            train_loss, train_mel, train_gate = self.train_epoch(epoch)
            
            # 2. Evaluate
            val_loss, val_mel, val_gate = self.evaluate()
            logger.info(
                f"🔬 Epoch {epoch+1} Validation | "
                f"Loss: {val_loss:.4f} (Mel: {val_mel:.4f}, Gate: {val_gate:.4f})"
            )
            
            # 3. Step scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # 4. Log to WandB
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_mel_loss": train_mel,
                    "train_gate_loss": train_gate,
                    "val_loss": val_loss,
                    "val_mel_loss": val_mel,
                    "val_gate_loss": val_gate,
                    "learning_rate": current_lr
                })
                
            # 5. Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = self.config.checkpoints_dir / "best_model.pt"
                logger.info(f"💾 Validation loss improved! Saving best model checkpoint to {checkpoint_path}")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": self.model.config
                }, checkpoint_path)
                
        # Save final model checkpoint
        final_path = self.config.checkpoints_dir / "final_model.pt"
        logger.info(f"💾 Saving final model checkpoint to {final_path}")
        torch.save({
            "epoch": self.epochs - 1,
            "model_state_dict": self.model.state_dict(),
            "config": self.model.config
        }, final_path)
        
        if self.use_wandb:
            wandb.finish()
            
        logger.info("🎉 SpokenDialogueTrainer training cycle completed successfully!")
