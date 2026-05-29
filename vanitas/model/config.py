from dataclasses import dataclass, field

@dataclass
class VanitasModelConfig:
    """Hyperparameters for the three streams and fusion gates of the Vanitas Architecture."""
    
    # Audio Feature Parameters
    mel_bins: int = 80               # Dimension of continuous mel-spectrogram input
    sample_rate: int = 16000
    hop_length: int = 160            # 10ms frame stride
    
    # 🔵 Stream 1: Perception Stream (Mamba-2 SSM)
    perception_dim: int = 512        # Hidden model dimension (d_model)
    perception_layers: int = 12      # Number of stacked Mamba-2 blocks
    perception_state_dim: int = 64   # SSM state dimension (d_state)
    perception_conv_dim: int = 4     # 1D causal convolution kernel size (d_conv)
    perception_expand: int = 2       # Channel expansion factor (expand)
    
    # 🔴 Fusion Gates (Learned MLPs)
    gate_hidden_dim: int = 256       # Hidden layer dimension for gates
    
    # 🟠 Stream 2: Cognition Core (Sparse Attention) - Phase 3
    cognition_dim: int = 512
    cognition_heads: int = 8
    cognition_layers: int = 8
    cognition_vocab_dim: int = 512   # Semantic embedding dimension
    memory_dim: int = 512            # Dimension of retrieved external facts
    
    # 🟢 Stream 3: Production Stream (Mamba-2 + Flow Matching) - Phase 4
    production_dim: int = 512
    production_layers: int = 12
    production_state_dim: int = 64
    
    # Training & Regularization
    dropout: float = 0.1
    adapter_rank: int = 8             # Zero-init adapters for customer/domain fine-tuning
