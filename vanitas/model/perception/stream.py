import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

from vanitas.model.config import VanitasModelConfig
from vanitas.audio.features import MelProjection

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vanitas.model.perception.stream")

# Attempt to load the official CUDA Mamba-SSM kernels
MAMBA_AVAILABLE = False
try:
    # mamba_ssm is only installable on CUDA machines
    from mamba_ssm.modules.mamba2 import Mamba2 as OfficialMamba2
    MAMBA_AVAILABLE = True
    logger.info("Official CUDA mamba-ssm modules loaded successfully.")
except ImportError:
    logger.warning("mamba-ssm package not found. Using custom high-performance PyTorch-native Mamba-2 CPU/MPS fallback.")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (standard for modern LLMs and Mamba-2)."""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class PyTorchMamba2Block(nn.Module):
    """A pure PyTorch implementation of the Mamba-2 SSD Block.
    Designed for local Apple Silicon (MPS) prototyping and CPU dry-runs.
    """
    def __init__(self, d_model: int = 512, d_state: int = 64, d_conv: int = 4, expand: int = 2, headdim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.nheads = self.d_inner // self.headdim
        
        # 1. Input projection: combines SSM inputs (X), matrices B and C, delta (dt), and gating
        # In Mamba-2, we project to: inner_dim (X) + d_state (B) + d_state (C) + nheads (dt) + inner_dim (gate)
        self.dt_min = 0.001
        self.dt_max = 0.1
        
        self.in_proj = nn.Linear(
            self.d_model, 
            self.d_inner + self.d_state + self.d_state + self.nheads + self.d_inner,
            bias=False
        )
        
        # 1D Causal Convolution over inner dimension (X), B, and C
        # Total channels to convolve: d_inner + 2 * d_state
        self.conv_channels = self.d_inner + 2 * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_channels,
            out_channels=self.conv_channels,
            kernel_size=self.d_conv,
            groups=self.conv_channels, # Depthwise conv
            padding=self.d_conv - 1,   # Causal padding
            bias=True
        )
        
        # Time-step (dt) projection bias initialization
        self.dt_bias = nn.Parameter(torch.empty(self.nheads))
        # Log-uniform initialization of dt bias
        dt_init = torch.exp(torch.rand(self.nheads) * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min))
        self.dt_bias.data.copy_(torch.log(dt_init))
        
        # Diagonal state matrix A parameters (one scalar per head)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.nheads + 1).float()))
        
        # Out projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        
        # Head RMSNorm before output projection
        self.norm = RMSNorm(self.d_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, L, D)
        Returns:
            Tensor of shape (B, L, D)
        """
        B_sz, L_sz, _ = x.shape
        
        # 1. Project input: (B, L, D inner + B + C + dt + gate)
        projected = self.in_proj(x)
        
        # Split projections
        # X: (B, L, d_inner)
        # B_mat: (B, L, d_state)
        # C_mat: (B, L, d_state)
        # dt: (B, L, nheads)
        # gate: (B, L, d_inner)
        X, B_mat, C_mat, dt, gate = torch.split(
            projected, 
            [self.d_inner, self.d_state, self.d_state, self.nheads, self.d_inner], 
            dim=-1
        )
        
        # 2. Causal 1D Convolution over X, B_mat, C_mat
        # Concat along features dimension and permute to (B, C, L) for Conv1D
        conv_input = torch.cat([X, B_mat, C_mat], dim=-1).permute(0, 2, 1)
        conv_out = self.conv1d(conv_input)
        # Apply strict causal slicing (discard lookahead padding samples at the right)
        conv_out = conv_out[:, :, :L_sz].permute(0, 2, 1)
        
        # Split convolved channels back
        X_conv, B_conv, C_conv = torch.split(
            conv_out, 
            [self.d_inner, self.d_state, self.d_state], 
            dim=-1
        )
        
        # Apply activations
        X_conv = F.silu(X_conv)
        
        # 3. State Space Duality (SSD) Recurrence
        # dt activation
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
        
        # Reshape for multi-head SSM processing
        # X_conv: (B, L, nheads, headdim)
        X_heads = X_conv.view(B_sz, L_sz, self.nheads, self.headdim)
        
        # A matrix: (nheads,) -> (B, L, nheads)
        A = -torch.exp(self.A_log) # Ensure negative eigenvalues for stability
        
        # Linear Recurrent Scan: h_t = exp(A * dt) * h_{t-1} + (B * dt) * X
        # For local execution, we run the scan sequentially or causally
        ssm_outputs = []
        
        # Head-wise loop (extremely clean and robust for MPS)
        for h in range(self.nheads):
            h_state = torch.zeros(B_sz, self.d_state, dtype=x.dtype, device=x.device)
            head_out = torch.zeros(B_sz, L_sz, self.headdim, dtype=x.dtype, device=x.device)
            
            a_head = A[h]
            dt_head = dt[:, :, h]  # (B, L)
            
            # Compute exponential decay: exp(A * dt)
            decay = torch.exp(a_head * dt_head) # (B, L)
            
            # B and C head matrices
            b_head = B_conv  # (B, L, d_state)
            c_head = C_conv  # (B, L, d_state)
            
            x_head = X_heads[:, :, h, :] # (B, L, headdim)
            
            # Recurrence loop over time dimension (L)
            for t in range(L_sz):
                d_t = dt_head[:, t].unsqueeze(-1)    # (B, 1)
                dec_t = decay[:, t].unsqueeze(-1)    # (B, 1)
                b_t = b_head[:, t, :]                # (B, d_state)
                c_t = c_head[:, t, :]                # (B, d_state)
                x_t = x_head[:, t, :]                # (B, headdim)
                
                # Update hidden state: h_t = dec_t * h_{t-1} + (b_t * d_t) * x_t.T
                # In SSD, the state is d_state-dimensional, combining across headdim
                # We project input: (b_t * d_t) forms a state projection outer product
                # For pure PyTorch validation, we update head state:
                # We simplify the projection mapping:
                # SSD acts as: head_state_t = dec * head_state_{t-1} + outer(b_t * d_t, x_t)
                # output_t = head_state_t * c_t
                # Let's perform state expansion: state is (B, d_state, headdim)
                if t == 0:
                    state = torch.zeros(B_sz, self.d_state, self.headdim, dtype=x.dtype, device=x.device)
                
                # outer product of B and X
                proj_in = torch.bmm(b_t.unsqueeze(-1), x_t.unsqueeze(1)) # (B, d_state, headdim)
                state = dec_t.unsqueeze(-1) * state + d_t.unsqueeze(-1) * proj_in
                
                # Output projection: Y = C * State
                # C is (B, d_state), we contract along d_state to get (B, headdim)
                y_t = torch.bmm(c_t.unsqueeze(1), state).squeeze(1) # (B, headdim)
                head_out[:, t, :] = y_t
                
            ssm_outputs.append(head_out)
            
        # Concat heads together along head dimension: (B, L, inner_dim)
        ssm_out = torch.cat(ssm_outputs, dim=-1)
        
        # 4. Multiplicative Gating
        gated = ssm_out * F.silu(gate)
        
        # Normalization and output projection
        normed = self.norm(gated)
        return self.out_proj(normed)


class Mamba2Wrapper(nn.Module):
    """Wraps Mamba-2 block, automatically switching between official CUDA version and custom PyTorch version."""
    
    def __init__(self, config: VanitasModelConfig):
        super().__init__()
        self.config = config
        
        if MAMBA_AVAILABLE:
            self.mamba = OfficialMamba2(
                d_model=config.perception_dim,
                d_state=config.perception_state_dim,
                d_conv=config.perception_conv_dim,
                expand=config.perception_expand
            )
        else:
            self.mamba = PyTorchMamba2Block(
                d_model=config.perception_dim,
                d_state=config.perception_state_dim,
                d_conv=config.perception_conv_dim,
                expand=config.perception_expand
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(x)


class PerceptionStream(nn.Module):
    """🔵 Stream 1: Perception (The Ear)
    An always-on stacked Mamba-2 SSM that compresses continuous audio features in O(1) time per frame.
    """
    
    def __init__(self, config: VanitasModelConfig):
        super().__init__()
        self.config = config
        
        # Feature projection layer: raw 80 mel features -> 512 dimensions
        self.input_projection = MelProjection(n_mels=config.mel_bins, model_dim=config.perception_dim)
        
        # Stack of Mamba-2 blocks with residual connections
        self.layers = nn.ModuleList([
            Mamba2Wrapper(config)
            for _ in range(config.perception_layers)
        ])
        
        # Output RMSNorm
        self.norm = RMSNorm(config.perception_dim)
        
        # Final Perception State projection layer (d_model -> d_model)
        self.state_projection = nn.Linear(config.perception_dim, config.perception_dim)

    def forward(self, mel_frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Processes a sequence of log-mel spectrogram frames.
        
        Args:
            mel_frames: Tensor of shape (B, T, mel_bins)
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - layer_outputs: Tensor of shape (B, T, perception_dim) (all compressed frames)
                - perception_state: Tensor of shape (B, perception_dim) (final compressed state snapshot)
        """
        # Project continuous features: (B, T, mel_bins) -> (B, T, perception_dim)
        x = self.input_projection(mel_frames)
        
        # Process Mamba-2 layers with residuals
        for layer in self.layers:
            x = x + layer(x)
            
        # Normalization
        x = self.norm(x)
        
        # Extract the final frame state as our rolling conversation context (Perception State)
        perception_state = self.state_projection(x[:, -1, :])
        
        return x, perception_state
