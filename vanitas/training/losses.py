import torch
import torch.nn as nn
import torch.nn.functional as F

class JointPerceptionGateLoss(nn.Module):
    """Calculates multi-task losses: Masked Mel Reconstruction MSE (Perception) + Weighted BCE (Fusion Gate)."""
    
    def __init__(self, mel_bins: int = 80, model_dim: int = 512, gate_pos_weight: float = 10.0):
        super().__init__()
        self.mel_bins = mel_bins
        self.model_dim = model_dim
        self.gate_pos_weight = gate_pos_weight
        
        # Self-supervised mel reconstruction head (maps 512 model dim -> 80 mel bins)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.model_dim, 256),
            nn.SiLU(),
            nn.Linear(256, self.mel_bins)
        )

    def forward(self, 
                perception_outputs: torch.Tensor, 
                think_gate_preds: torch.Tensor, 
                ground_truth_mel: torch.Tensor, 
                mask_indices: torch.Tensor, 
                turn_targets: torch.Tensor,
                lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates joint losses with sequence padding masking.
        
        Args:
            perception_outputs: (B, T, model_dim)
            think_gate_preds: (B, T, 1)
            ground_truth_mel: (B, T, mel_bins)
            mask_indices: (B, T) (1=masked frame)
            turn_targets: (B, T, 1) (1=turn boundary)
            lengths: (B,) (actual unpadded sequence lengths)
            
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - total_loss: Weighted sum of both losses.
                - mel_loss: Masked mel reconstruction MSE loss.
                - gate_loss: Weighted turn gate BCE loss.
        """
        B_sz, T_sz, _ = perception_outputs.shape
        device = perception_outputs.device
        
        # Create sequence length mask to ignore padded tails
        seq_mask = torch.zeros(B_sz, T_sz, device=device)
        for i, length in enumerate(lengths):
            seq_mask[i, :length] = 1.0
            
        # 1. Masked Mel Spectrogram Reconstruction Loss (MSE)
        # Project Perception hidden states back to mel space
        pred_mel = self.reconstruction_head(perception_outputs) # (B, T, mel_bins)
        
        # Compute squared error: (B, T, mel_bins)
        squared_err = (pred_mel - ground_truth_mel).pow(2)
        
        # Apply mask_indices and sequence length mask
        # mask_indices is (B, T), unsqueeze to (B, T, 1)
        mel_active_mask = mask_indices.unsqueeze(-1) * seq_mask.unsqueeze(-1)
        
        # Sum squared error over active masked frames and divide by number of masked frames
        masked_sum = (squared_err * mel_active_mask).sum()
        total_masked_elements = mel_active_mask.sum() * self.mel_bins
        
        # Prevent division by zero
        mel_loss = masked_sum / max(1.0, total_masked_elements.item())
        
        # 2. Gating turn-taking classification loss (Weighted BCE)
        # Clip probabilities to avoid log(0) instabilities
        think_preds_clipped = torch.clamp(think_gate_preds, min=1e-6, max=1.0 - 1e-6)
        
        # Weighted BCE calculation:
        # loss = - [w * y * log(p) + (1 - y) * log(1 - p)]
        pos_weight = self.gate_pos_weight
        bce_loss_raw = - (
            pos_weight * turn_targets * torch.log(think_preds_clipped) + 
            (1.0 - turn_targets) * torch.log(1.0 - think_preds_clipped)
        ) # (B, T, 1)
        
        # Apply sequence length mask
        gate_active_mask = seq_mask.unsqueeze(-1)
        gate_loss_sum = (bce_loss_raw * gate_active_mask).sum()
        total_gate_elements = gate_active_mask.sum()
        
        gate_loss = gate_loss_sum / max(1.0, total_gate_elements.item())
        
        # 3. Scheduled Combined Loss
        # Lambdas balance the scale of MSE (~0.5) and weighted BCE (~0.2)
        lambda_mel = 1.0
        lambda_gate = 2.0
        
        total_loss = lambda_mel * mel_loss + lambda_gate * gate_loss
        
        return total_loss, mel_loss, gate_loss
