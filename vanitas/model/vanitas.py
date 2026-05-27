import torch
import torch.nn as nn

from vanitas.model.config import VanitasModelConfig
from vanitas.model.perception.stream import PerceptionStream
from vanitas.model.gates.fusion import FusionGateBank
from vanitas.model.cognition.core import CognitionCore
from vanitas.model.production.stream import ProductionStream
from vanitas.model.production.flow_head import FlowMatchingHead
from vanitas.model.production.vocoder import HiFiGANWrapper

class VanitasModel(nn.Module):
    """The master Vanitas Architecture model unifying all three streams and learned fusion gates."""
    
    def __init__(self, config: VanitasModelConfig = None):
        super().__init__()
        self.config = config if config is not None else VanitasModelConfig()
        
        # 🔵 Stream 1: Always-On Perception Stream
        self.perception = PerceptionStream(self.config)
        
        # 🔴 Learned Fusion Gates Bank
        self.gates = FusionGateBank(self.config)
        
        # 🟠 Stream 2: Cognition Core (The Brain)
        self.cognition = CognitionCore(self.config)
        
        # 🟢 Stream 3: Production Stream (The Mouth)
        self.production_stream = ProductionStream(
            d_model=self.config.production_dim, 
            d_state=self.config.production_state_dim, 
            n_layers=self.config.production_layers
        )
        self.flow_head = FlowMatchingHead(
            d_model=self.config.production_dim, 
            mel_dim=self.config.mel_bins
        )
        self.vocoder = HiFiGANWrapper(
            mel_dim=self.config.mel_bins, 
            hop_length=self.config.hop_length
        )

    def forward(self, mel_frames: torch.Tensor, memory_embeddings: torch.Tensor = None, time_steps: torch.Tensor = None) -> tuple:
        """
        Processes continuous mel frames streamingly through the three-stream architecture.
        
        Args:
            mel_frames: Tensor of shape (B, T, mel_bins)
            memory_embeddings: Optional tensor of shape (B, K, memory_dim) for Cognition
            time_steps: Optional tensor of shape (B, 1) for Flow Matching training
            
        Returns:
            tuple containing:
                - audio_waveforms: (B, 1, T * hop_length) Output audio via vocoder (inference only)
                - v_pred: (B, T, mel_dim) Flow matching velocity (training only)
                - gates: Tuple of (think_gate, backchannel_gate, speak_gate)
        """
        # 1. Run through Mamba-2 Perception Stream
        perception_outputs, perception_state = self.perception(mel_frames)
        
        # 2. Run outputs frame-by-frame through Fusion Gates
        think_gate, backchannel_gate, speak_gate = self.gates(perception_outputs)
        
        # 3. Trigger Cognition Core (Reasoning)
        cognition_state = self.cognition(perception_outputs, memory_embeddings)
        
        # 4. Run through Production Stream (Generation)
        # Align dimensions if necessary (here they are both 512)
        production_state = self.production_stream(cognition_state, perception_outputs)
        
        # 5. Flow Matching generation
        if self.training and time_steps is not None:
            # Training mode: Predict vector fields
            v_pred = self.flow_head(production_state, time_steps)
            return None, v_pred, (think_gate, backchannel_gate, speak_gate)
        else:
            # Inference mode: Euler integration and Vocoder
            mel_pred = self.flow_head.generate(production_state, steps=10)
            audio_waveforms = self.vocoder(mel_pred)
            return audio_waveforms, None, (think_gate, backchannel_gate, speak_gate)
