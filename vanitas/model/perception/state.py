import torch
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class PerceptionStateCache:
    """A thread-safe cache containing the recurrent activation history of our stacked Mamba-2 stream."""
    
    # Map from layer index to convolutional buffer state of shape (B, conv_channels, d_conv - 1)
    conv_states: Dict[int, torch.Tensor] = field(default_factory=dict)
    
    # Map from layer index to SSM recurrent state of shape (B, nheads, d_state, headdim)
    ssm_states: Dict[int, torch.Tensor] = field(default_factory=dict)
    
    def reset(self):
        """Clears all recurrent history in the state cache."""
        self.conv_states.clear()
        self.ssm_states.clear()

    def clone(self) -> 'PerceptionStateCache':
        """Creates a deep copy of the active cache tensors."""
        cloned = PerceptionStateCache()
        for k, v in self.conv_states.items():
            cloned.conv_states[k] = v.clone()
        for k, v in self.ssm_states.items():
            cloned.ssm_states[k] = v.clone()
        return cloned

    def to(self, device: torch.device | str) -> 'PerceptionStateCache':
        """Moves all state tensors in the cache to the target computation device."""
        for k, v in self.conv_states.items():
            self.conv_states[k] = v.to(device)
        for k, v in self.ssm_states.items():
            self.ssm_states[k] = v.to(device)
        return self
