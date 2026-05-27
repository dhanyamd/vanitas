import torch
import torch.nn as nn
import torch.nn.functional as F

from vanitas.model.config import VanitasModelConfig

class FusionGate(nn.Module):
    """A learned neural gate MLP that processes Perception States to output a gating activation signal [0, 1]."""
    
    def __init__(self, state_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Restricts the gate scalar output strictly to [0, 1]
        )
        
    def forward(self, perception_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            perception_state: Tensor of shape (B, D) or (B, T, D)
        Returns:
            Tensor of shape (B, 1) or (B, T, 1) containing gate probabilities.
        """
        return self.net(perception_state)


class FusionGateBank(nn.Module):
    """The bank containing all learned gates controlling cross-stream information flows."""
    
    def __init__(self, config: VanitasModelConfig):
        super().__init__()
        self.config = config
        
        # 1. Gate P→C: Perception to Cognition (When should we think? Replaces VAD/endpointing)
        self.gate_pc = FusionGate(config.perception_dim, config.gate_hidden_dim)
        
        # 2. Gate P→G: Perception to Production (Direct backchanneling bypass - laughter, sighs, mm-hmm)
        self.gate_pg = FusionGate(config.perception_dim, config.gate_hidden_dim)
        
        # 3. Gate C→G: Cognition to Production (When is a response fully planned and ready to release?)
        self.gate_cg = FusionGate(config.cognition_dim, config.gate_hidden_dim)

    def forward(self, perception_states: torch.Tensor, cognition_states: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates all gates in parallel.
        
        Args:
            perception_states: Tensor of shape (B, T, D_perceive) or (B, D_perceive)
            cognition_states: Optional tensor of shape (B, T, D_cognition) or (B, D_cognition)
            
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - think_gate: Gating signal [0, 1] for Perception -> Cognition activation.
                - backchannel_gate: Gating signal [0, 1] for direct Perception -> Production bypass.
                - speak_gate: Gating signal [0, 1] for Cognition -> Production execution.
        """
        think_gate = self.gate_pc(perception_states)
        backchannel_gate = self.gate_pg(perception_states)
        
        if cognition_states is not None:
            speak_gate = self.gate_cg(cognition_states)
        else:
            # Fallback zero tensor if cognition has not yet activated
            if len(perception_states.shape) == 3:
                speak_gate = torch.zeros(perception_states.shape[0], perception_states.shape[1], 1, device=perception_states.device)
            else:
                speak_gate = torch.zeros(perception_states.shape[0], 1, device=perception_states.device)
                
        return think_gate, backchannel_gate, speak_gate
