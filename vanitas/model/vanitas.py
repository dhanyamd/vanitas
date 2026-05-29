import torch
import torch.nn as nn
from dataclasses import fields

from vanitas.model.config import VanitasModelConfig
from vanitas.model.adapters import LowRankAdapter
from vanitas.model.perception.stream import PerceptionStream
from vanitas.model.gates.fusion import FusionGateBank
from vanitas.model.cognition.core import CognitionCore
from vanitas.model.cognition.speech_memory import SpeechMemoryEncoder
from vanitas.model.production.stream import ProductionStream
from vanitas.model.production.flow_head import FlowMatchingHead
from vanitas.model.production.vocoder import HiFiGANWrapper

class VanitasModel(nn.Module):
    """The master Vanitas Architecture model unifying all three streams and learned fusion gates."""
    
    def __init__(self, config: VanitasModelConfig = None):
        super().__init__()
        self.config = self._normalize_config(config)
        
        # 🔵 Stream 1: Always-On Perception Stream
        self.perception = PerceptionStream(self.config)
        self.perception_adapter = LowRankAdapter(self.config.perception_dim, rank=self.config.adapter_rank)
        
        # 🔴 Learned Fusion Gates Bank
        self.gates = FusionGateBank(self.config)
        
        # 🟠 Stream 2: Cognition Core (The Brain)
        self.cognition = CognitionCore(self.config)
        self.cognition_adapter = LowRankAdapter(self.config.cognition_dim, rank=self.config.adapter_rank)
        self.speech_memory_encoder = SpeechMemoryEncoder(
            mel_dim=self.config.mel_bins,
            memory_dim=self.config.cognition_dim,
            hidden_dim=self.config.cognition_dim,
        )
        
        # 🟢 Stream 3: Production Stream (The Mouth)
        self.production_stream = ProductionStream(
            d_model=self.config.production_dim, 
            d_state=self.config.production_state_dim, 
            n_layers=self.config.production_layers
        )
        self.production_adapter = LowRankAdapter(self.config.production_dim, rank=self.config.adapter_rank)
        self.flow_head = FlowMatchingHead(
            d_model=self.config.production_dim, 
            mel_dim=self.config.mel_bins
        )
        self.vocoder = HiFiGANWrapper(
            mel_dim=self.config.mel_bins, 
            hop_length=self.config.hop_length
        )

    @staticmethod
    def _normalize_config(config: VanitasModelConfig | dict | None) -> VanitasModelConfig:
        defaults = VanitasModelConfig()
        if config is None:
            return defaults
        if isinstance(config, dict):
            source = config
        else:
            source = {field.name: getattr(config, field.name) for field in fields(defaults) if hasattr(config, field.name)}
        merged = {field.name: source.get(field.name, getattr(defaults, field.name)) for field in fields(defaults)}
        return VanitasModelConfig(**merged)

    def forward(
        self,
        mel_frames: torch.Tensor,
        agent_mel_frames: torch.Tensor = None,
        memory_embeddings: torch.Tensor = None,
        time_steps: torch.Tensor = None,
        noisy_mel: torch.Tensor = None,
        target_agent_mel: torch.Tensor = None,
        target_agent_audio: torch.Tensor = None,
        prosody_features: torch.Tensor = None,
        lengths: torch.Tensor = None,
    ) -> dict:
        """
        Processes continuous mel frames streamingly through the three-stream architecture.
        
        Args:
            mel_frames: Tensor of shape (B, T, mel_bins)
            agent_mel_frames: Optional tensor of shape (B, T, mel_bins) for Duplex feedback
            memory_embeddings: Optional tensor of shape (B, K, memory_dim) for Cognition
            time_steps: Optional tensor of shape (B, 1) for Flow Matching training
            noisy_mel: Optional tensor of shape (B, T, mel_bins) for current flow state y_t
            target_agent_mel: Optional tensor of shape (B, T, mel_bins) for speech-memory training target
            target_agent_audio: Optional tensor of shape (B, 1, T * hop_length) for vocoder training
            prosody_features: Optional tensor of shape (B, 8) for speech-memory conditioning
            lengths: Optional tensor of shape (B,) with non-padded sequence lengths
            
        Returns:
            dict containing:
                - "audio": (B, 1, T * hop_length) Output audio via vocoder (inference only)
                - "v_pred": (B, T, mel_dim) Flow matching velocity (training only)
                - "perception_outputs": (B, T, perception_dim) Perception stream hidden outputs
                - "perception_state": (B, perception_dim) Final state snapshot of Perception
                - "cognition_state": (B, T, cognition_dim) Reasoning state from Cognition Core
                - "think_gate": (B, T, 1) Gating activation for turn-taking
                - "backchannel_gate": (B, T, 1) Gating activation for backchannels
                - "speak_gate": (B, T, 1) Gating activation for speaking
                - "speech_memory_target": Optional (B, cognition_dim) speech-native memory target
                - "vocoder_audio": Optional (B, 1, T * hop_length) reconstruction from target_agent_mel
        """
        # 1. Run through Mamba-2 Perception Stream
        perception_outputs, perception_state = self.perception(mel_frames, agent_mel_frames)
        perception_outputs = self.perception_adapter(perception_outputs)
        perception_state = perception_outputs[:, -1, :]
        
        # 2. Trigger Cognition Core (Reasoning)
        cognition_state = self.cognition(perception_outputs, memory_embeddings)
        cognition_state = self.cognition_adapter(cognition_state)
        
        # 3. Run outputs frame-by-frame through Fusion Gates
        think_gate, backchannel_gate, speak_gate = self.gates(perception_outputs, cognition_state)
        
        # 4. Run through Production Stream (Generation)
        # Align dimensions if necessary (here they are both 512)
        production_state = self.production_stream(cognition_state, perception_outputs)
        production_state = self.production_adapter(production_state)

        speech_memory_target = None
        vocoder_audio = None
        if target_agent_mel is not None:
            speech_memory_target = self.speech_memory_encoder(
                target_agent_mel,
                lengths=lengths,
                prosody_features=prosody_features,
            )
            if self.training and target_agent_audio is not None:
                vocoder_audio = self.vocoder(target_agent_mel)
        
        # 5. Flow Matching generation
        if self.training and time_steps is not None:
            # Training mode: Predict vector fields
            v_pred = self.flow_head(production_state, time_steps, noisy_mel)
            audio_waveforms = None
        else:
            # Inference mode: Euler integration and Vocoder
            v_pred = None
            mel_pred = self.flow_head.generate(production_state, steps=10)
            audio_waveforms = self.vocoder(mel_pred)
            
        return {
            "audio": audio_waveforms,
            "v_pred": v_pred,
            "perception_outputs": perception_outputs,
            "perception_state": perception_state,
            "cognition_state": cognition_state,
            "think_gate": think_gate,
            "backchannel_gate": backchannel_gate,
            "speak_gate": speak_gate,
            "speech_memory_target": speech_memory_target,
            "vocoder_audio": vocoder_audio,
        }
