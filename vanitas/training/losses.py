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


class FullVanitasLoss(nn.Module):
    """Calculates the complete joint loss for all three streams:
    1. Perception Self-Supervised Masked Mel Reconstruction (MSE)
    2. Fusion Gates Gating (BCE on turn-taking triggers)
    3. Cognition Core Context Alignment (Symmetric CLIP-style Contrastive InfoNCE)
    4. Production Stream continuous Mel generation (Flow Matching MSE)
    """
    
    def __init__(self, 
                 mel_bins: int = 80, 
                 model_dim: int = 512, 
                 gate_pos_weight: float = 10.0, 
                 contrastive_temp: float = 0.07):
        super().__init__()
        self.mel_bins = mel_bins
        self.model_dim = model_dim
        self.gate_pos_weight = gate_pos_weight
        self.contrastive_temp = contrastive_temp
        
        # Self-supervised mel reconstruction head (maps 512 model dim -> 80 mel bins)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.model_dim, 256),
            nn.SiLU(),
            nn.Linear(256, self.mel_bins)
        )
        
    def forward(self, 
                model_outputs: dict, 
                ground_truth_mel: torch.Tensor,     # (B, T, mel_bins) (User mel input)
                mask_indices: torch.Tensor,         # (B, T)
                turn_targets: torch.Tensor,         # (B, T, 1)
                agent_mel: torch.Tensor,            # (B, T, mel_bins) (Agent target mel)
                semantic_target: torch.Tensor,      # (B, D_memory) (Target semantic embedding)
                lengths: torch.Tensor,              # (B,) (actual sequence lengths)
                flow_target: torch.Tensor = None,
                speak_targets: torch.Tensor = None,
                agent_active_mask: torch.Tensor = None,
                agent_audio: torch.Tensor = None) -> dict:
        """
        Computes the unified joint loss.
        """
        device = ground_truth_mel.device
        B_sz, T_sz, _ = ground_truth_mel.shape
        
        # Create sequence length mask to ignore padded tails
        seq_mask = torch.zeros(B_sz, T_sz, device=device)
        for i, length in enumerate(lengths):
            seq_mask[i, :length] = 1.0
            
        # 1. Masked Mel Spectrogram Reconstruction Loss (MSE) - Perception Stream
        perception_outputs = model_outputs["perception_outputs"]
        pred_mel = self.reconstruction_head(perception_outputs)
        squared_err = (pred_mel - ground_truth_mel).pow(2)
        mel_active_mask = mask_indices.unsqueeze(-1) * seq_mask.unsqueeze(-1)
        masked_sum = (squared_err * mel_active_mask).sum()
        total_masked_elements = mel_active_mask.sum() * self.mel_bins
        mel_loss = masked_sum / max(1.0, total_masked_elements.item())
        
        # 2. Gating Turn-Taking Classification Loss (Weighted BCE) - Fusion Gates
        think_gate_preds = model_outputs["think_gate"]
        speak_gate_preds = model_outputs["speak_gate"]
        if speak_targets is None:
            speak_targets = turn_targets
        think_preds_clipped = torch.clamp(think_gate_preds, min=1e-6, max=1.0 - 1e-6)
        speak_preds_clipped = torch.clamp(speak_gate_preds, min=1e-6, max=1.0 - 1e-6)
        pos_weight = self.gate_pos_weight
        think_bce_loss_raw = - (
            pos_weight * turn_targets * torch.log(think_preds_clipped) + 
            (1.0 - turn_targets) * torch.log(1.0 - think_preds_clipped)
        )
        speak_bce_loss_raw = - (
            pos_weight * speak_targets * torch.log(speak_preds_clipped) + 
            (1.0 - speak_targets) * torch.log(1.0 - speak_preds_clipped)
        )
        gate_active_mask = seq_mask.unsqueeze(-1)
        gate_loss_sum = ((think_bce_loss_raw + speak_bce_loss_raw) * gate_active_mask).sum()
        total_gate_elements = gate_active_mask.sum()
        gate_loss = gate_loss_sum / max(1.0, 2.0 * total_gate_elements.item())
        
        # 3. Cognition Context Alignment Loss (Symmetric CLIP InfoNCE) - Cognition Core
        # Pool cognition states at turn boundary frames (where turn_targets == 1)
        cognition_state = model_outputs["cognition_state"] # (B, T, D_cognition)
        boundary_weights = turn_targets * seq_mask.unsqueeze(-1) # (B, T, 1)
        sum_rep = (cognition_state * boundary_weights).sum(dim=1) # (B, D_cognition)
        sum_weight = boundary_weights.sum(dim=1) # (B, 1)
        
        active_rep = sum_rep / torch.clamp(sum_weight, min=1e-6) # (B, D_cognition)
        mean_rep = (cognition_state * seq_mask.unsqueeze(-1)).sum(dim=1) / torch.clamp(seq_mask.sum(dim=1, keepdim=True), min=1e-6)
        
        # If a sequence has no boundary triggers, fall back to average representation
        has_boundary = (sum_weight > 0).float() # (B, 1)
        pooled_rep = has_boundary * active_rep + (1.0 - has_boundary) * mean_rep # (B, D_cognition)
        
        speech_memory_target = model_outputs.get("speech_memory_target")
        if speech_memory_target is not None and speech_memory_target.shape[-1] == pooled_rep.shape[-1]:
            # Speech-native memory is the main target. The lexical vector is a
            # lightweight teacher/fallback that prevents the space from becoming
            # purely acoustic when text metadata is available.
            if semantic_target.shape[-1] == speech_memory_target.shape[-1]:
                semantic_target = F.normalize(0.8 * speech_memory_target + 0.2 * semantic_target, p=2, dim=-1)
            else:
                semantic_target = speech_memory_target

        # Normalize representations to compute cosine similarities
        pooled_rep_norm = F.normalize(pooled_rep, p=2, dim=-1)
        semantic_target_norm = F.normalize(semantic_target, p=2, dim=-1)
        
        # Cosine similarity matrix: (B, B)
        similarity_matrix = torch.matmul(pooled_rep_norm, semantic_target_norm.t()) / self.contrastive_temp
        labels = torch.arange(B_sz, device=device)
        
        loss_queries = F.cross_entropy(similarity_matrix, labels)
        loss_targets = F.cross_entropy(similarity_matrix.t(), labels)
        cognition_loss = (loss_queries + loss_targets) / 2.0
        
        # 4. Production Continuous Flow Matching Loss - Production Stream
        v_pred = model_outputs["v_pred"] # (B, T, mel_bins)
        if flow_target is None:
            flow_target = agent_mel
        if agent_active_mask is None:
            agent_active_mask = torch.ones(B_sz, T_sz, 1, device=device)

        # Weight production toward real agent speech. Silence is still learned
        # with a small weight, but cannot dominate the velocity field.
        flow_squared_err = (v_pred - flow_target).pow(2)
        flow_active_mask = seq_mask.unsqueeze(-1) * (0.1 + 0.9 * agent_active_mask)
        flow_loss_sum = (flow_squared_err * flow_active_mask).sum()
        total_flow_elements = flow_active_mask.sum() * self.mel_bins
        flow_loss = flow_loss_sum / max(1.0, total_flow_elements.item())

        # 5. Trainable vocoder waveform reconstruction loss.
        vocoder_audio = model_outputs.get("vocoder_audio")
        if vocoder_audio is not None and agent_audio is not None:
            min_samples = min(vocoder_audio.shape[-1], agent_audio.shape[-1])
            pred_audio = vocoder_audio[..., :min_samples]
            target_audio = agent_audio[..., :min_samples]
            vocoder_loss = F.l1_loss(pred_audio, target_audio)
        else:
            vocoder_loss = torch.zeros((), device=device)
        
        # 6. Joint Loss Weighted Combination
        lambda_mel = 1.0
        lambda_gate = 2.0
        lambda_cognition = 1.0
        lambda_flow = 1.0
        lambda_vocoder = 0.25
        
        total_loss = (
            lambda_mel * mel_loss + 
            lambda_gate * gate_loss + 
            lambda_cognition * cognition_loss + 
            lambda_flow * flow_loss +
            lambda_vocoder * vocoder_loss
        )
        
        return {
            "total_loss": total_loss,
            "mel_loss": mel_loss,
            "gate_loss": gate_loss,
            "cognition_loss": cognition_loss,
            "flow_loss": flow_loss,
            "vocoder_loss": vocoder_loss,
        }
