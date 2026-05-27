import torch
import torch.nn as nn
from vanitas.model.perception.stream import Mamba2Wrapper
from vanitas.model.config import VanitasModelConfig

class ProductionStream(nn.Module):
    """
    Production Stream (The Mouth).
    Uses Mamba-2 to model continuous generation based on Cognition output.
    """
    def __init__(self, d_model: int, d_state: int = 128, n_layers: int = 4):
        super().__init__()
        self.d_model = d_model
        
        # We pass a minimal config for the Mamba wrapper to function
        config = VanitasModelConfig(perception_dim=d_model, perception_state_dim=d_state)
        self.layers = nn.ModuleList([
            Mamba2Wrapper(config)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, cognition_state: torch.Tensor, perception_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cognition_state: (B, L, D) - output from cognition core
            perception_state: (B, L, D) - output from perception stream
            
        Returns:
            production_state: (B, L, D) - conditioned state for the Flow Head
        """
        # Fuse cognition and perception states
        x = cognition_state + perception_state
        
        for layer in self.layers:
            x = layer(x)
            
        return self.norm(x)
