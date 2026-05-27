import torch
import torch.nn as nn
import torch.nn.functional as F

from vanitas.model.config import VanitasModelConfig

class CognitionAttentionBlock(nn.Module):
    """Multi-Head Self/Cross Attention Block specifically designed for reasoning over memory."""
    
    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Multi-Head Attention layer
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network (SwiGLU-inspired FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """Runs attention step.
        
        Args:
            q: Queries tensor of shape (B, Target_Len, d_model)
            kv: Keys/Values tensor of shape (B, Source_Len, d_model)
            
        Returns:
            Tensor of shape (B, Target_Len, d_model)
        """
        # Multi-Head Attention forward: (output, weights)
        attn_out, _ = self.attn(
            query=q,
            key=kv,
            value=kv
        )
        
        # Residual and LayerNorm 1
        x = self.norm1(q + self.dropout(attn_out))
        
        # FFN and LayerNorm 2
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class CognitionCore(nn.Module):
    """🟠 Stream 2: Cognition Core (The Brain)
    Stacked Cross-Attention blocks that fuse Perception States and retrieved Memory contexts to output a Cognition State.
    """
    
    def __init__(self, config: VanitasModelConfig):
        super().__init__()
        self.config = config
        
        # Projections to align dimensions if they differ
        self.perception_proj = nn.Linear(config.perception_dim, config.cognition_dim)
        self.memory_proj = nn.Linear(config.memory_dim, config.cognition_dim)
        
        # Positional embeddings for memory tokens to preserve factual ordering
        self.pos_encoder = nn.Parameter(torch.randn(1, 20, config.cognition_dim) * 0.02) # Supports up to 20 memory chunks
        
        # Stacked reasoning blocks
        self.layers = nn.ModuleList([
            CognitionAttentionBlock(
                d_model=config.cognition_dim,
                n_heads=config.cognition_heads,
                dropout=config.dropout
            )
            for _ in range(config.cognition_layers)
        ])
        
        # Output project head
        self.output_norm = nn.LayerNorm(config.cognition_dim)

    def forward(self, perception_state: torch.Tensor, memory_embeddings: torch.Tensor = None) -> torch.Tensor:
        """Reason over perception inputs and retrieved facts.
        
        Args:
            perception_state: Tensor of shape (B, perception_dim) or (B, T, perception_dim)
            memory_embeddings: Optional RAG context tensor of shape (B, K, memory_dim)
            
        Returns:
            torch.Tensor: Cognition State tensor of shape (B, cognition_dim) or (B, T, cognition_dim)
        """
        # Ensure correct query dimensions: (B, T, D_cognition)
        is_batched_flat = len(perception_state.shape) == 2
        if is_batched_flat:
            # Add virtual time dimension: (B, 1, perception_dim)
            q = perception_state.unsqueeze(1)
        else:
            q = perception_state
            
        # Project inputs to shared cognition dimension
        q = self.perception_proj(q)
        
        B_sz, T_sz, _ = q.shape
        
        # Handle memory routing
        if memory_embeddings is not None and memory_embeddings.shape[1] > 0:
            kv = self.memory_proj(memory_embeddings) # (B, K, d_cognition)
            
            # Apply positional embeddings to the memory sequence to preserve relative context order
            K_sz = kv.shape[1]
            kv = kv + self.pos_encoder[:, :K_sz, :]
        else:
            # Fallback self-attention if memory is empty
            kv = q
            
        # Propagate through stacked attention blocks
        x = q
        for layer in self.layers:
            # Query attends to Memory contexts (Cross-Attention)
            x = layer(x, kv)
            
        x = self.output_norm(x)
        
        if is_batched_flat:
            x = x.squeeze(1) # Convert back to (B, cognition_dim)
            
        return x
