import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowMatchingHead(nn.Module):
    """
    Continuous Flow Matching Head for generating Mel Spectrograms.
    Avoids discrete tokens, directly learning vector fields for continuous audio features.
    """
    def __init__(self, d_model: int, mel_dim: int = 80):
        super().__init__()
        self.d_model = d_model
        self.mel_dim = mel_dim
        
        # MLP to embed time steps
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Predicts the velocity field
        self.velocity_net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, mel_dim)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass.
        Args:
            x: (B, L, D) - Production stream state
            t: (B, 1) - Timesteps between [0, 1]
            
        Returns:
            v_pred: (B, L, mel_dim) - Predicted velocity
        """
        # Condition on time
        t_emb = self.time_mlp(t).unsqueeze(1) # (B, 1, D)
        x_cond = x + t_emb
        
        # Predict velocity
        v_pred = self.velocity_net(x_cond) 
        
        return v_pred
        
    def generate(self, x: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """
        Inference generation via Euler integration.
        Args:
            x: (B, L, D) - Production stream state
            steps: Number of integration steps
            
        Returns:
            mel: (B, L, mel_dim) - Generated Mel spectrogram
        """
        B, L, D = x.shape
        device = x.device
        
        # Start from standard Gaussian noise
        y = torch.randn(B, L, self.mel_dim, device=device)
        
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.ones(B, 1, device=device) * (i * dt)
            v_pred = self(x, t)
            y = y + v_pred * dt
            
        return y
