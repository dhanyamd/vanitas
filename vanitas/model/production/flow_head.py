import torch
import torch.nn as nn
import torch.nn.functional as F
import os

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
        self.mel_mlp = nn.Sequential(
            nn.Linear(mel_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        nn.init.zeros_(self.mel_mlp[-1].weight)
        nn.init.zeros_(self.mel_mlp[-1].bias)
        
        # Predicts the velocity field
        self.velocity_net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, mel_dim)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y_t: torch.Tensor = None) -> torch.Tensor:
        """
        Training forward pass.
        Args:
            x: (B, L, D) - Production stream state
            t: (B, 1) - Timesteps between [0, 1]
            y_t: (B, L, mel_dim) - Current noisy mel state
            
        Returns:
            v_pred: (B, L, mel_dim) - Predicted velocity
        """
        if y_t is None:
            y_t = torch.zeros(x.shape[0], x.shape[1], self.mel_dim, device=x.device, dtype=x.dtype)

        # Condition on time and current noisy mel state. Without y_t the
        # velocity target y1 - y0 is random from the model's point of view.
        t_emb = self.time_mlp(t).unsqueeze(1) # (B, 1, D)
        y_emb = self.mel_mlp(y_t)
        x_cond = x + t_emb + y_emb
        
        # Predict velocity
        v_pred = self.velocity_net(x_cond) 
        
        return v_pred
        
    def generate(self, x: torch.Tensor, steps: int = 8) -> torch.Tensor:
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
        
        # Deterministic zero-start decoding is safer for live inference and
        # keeps older checkpoints from exposing uncancelled Gaussian noise.
        # For stochastic CFM sampling, set VANITAS_FLOW_NOISE_SCALE=1.
        noise_scale = float(os.environ.get("VANITAS_FLOW_NOISE_SCALE", "0.0"))
        if noise_scale > 0.0:
            y = torch.randn(B, L, self.mel_dim, device=device) * noise_scale
        else:
            y = torch.zeros(B, L, self.mel_dim, device=device)
        
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.ones(B, 1, device=device) * (i * dt)
            v_pred = self(x, t, y)
            y = y + v_pred * dt

        # Keep generated mel values in the same rough dynamic range as training
        # extractor output (log10 power usually around [-5, ~2]).
        return torch.clamp(y, min=-5.0, max=2.0)
