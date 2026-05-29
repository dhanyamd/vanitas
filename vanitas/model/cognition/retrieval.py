import torch
import torch.nn as nn
import logging

from vanitas.model.config import VanitasModelConfig
from vanitas.model.cognition.memory import VectorMemory
from vanitas.model.cognition.text_embedding import lexical_text_embedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vanitas.model.cognition.retrieval")

class FactualRetriever(nn.Module):
    """Bridges the Vector Database and the PyTorch Cognition Core.
    Queries memory using continuous Perception States and encodes retrieved text facts back into tensors.
    """
    
    def __init__(self, config: VanitasModelConfig, memory: VectorMemory = None):
        super().__init__()
        self.config = config
        self.memory = memory if memory is not None else VectorMemory(embedding_dim=config.memory_dim)
        
        # Factual projection layer to align retrieved memory embeddings to target memory dimension
        self.output_proj = nn.Sequential(
            nn.Linear(config.memory_dim, config.memory_dim),
            nn.LayerNorm(config.memory_dim),
            nn.SiLU()
        )
        
        # Cache for hash-generated mock vector embeddings to avoid re-generating
        self._embedding_cache = {}

    def _deterministic_text_embedding(self, text: str) -> torch.Tensor:
        """Dependency-free lexical embedding for retrieved memory documents."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        tensor_vec = lexical_text_embedding(text, embedding_dim=self.config.memory_dim)
        self._embedding_cache[text] = tensor_vec
        return tensor_vec

    def forward(self, query_state: torch.Tensor, top_k: int = 3) -> tuple[torch.Tensor, list[list[str]]]:
        """Queries database using continuous queries, encodes matching documents, and projects them.
        
        Args:
            query_state: Query tensor of shape (B, D_perceive) (Perception State snapshot)
            top_k: Number of relevant context documents to fetch.
            
        Returns:
            tuple[torch.Tensor, list[list[str]]]:
                - memory_embeddings: Tensor of shape (B, K, D_memory) representing RAG context.
                - retrieved_documents: Nested list containing retrieved fact strings for auditing.
        """
        B_sz = query_state.shape[0]
        device = query_state.device
        
        # Convert torch query tensor to numpy for ChromaDB query
        query_np = query_state.detach().cpu().float().numpy() # (B, D_perceive)
        
        batch_docs = []
        batch_embeddings = []
        
        # Query database row-by-row in batch
        for b in range(B_sz):
            q_vec = query_np[b]
            matched = self.memory.query_semantic(q_vec, top_k=top_k)
            
            docs = []
            embs = []
            for item in matched:
                doc_text = item["document"]
                docs.append(doc_text)
                
                # Get deterministic embedding vector
                vec = self._deterministic_text_embedding(doc_text)
                embs.append(vec)
                
            batch_docs.append(docs)
            
            if embs:
                # Stack matching embeddings: (K, D_memory)
                stacked_embs = torch.stack(embs, dim=0).to(device)
            else:
                # Zero fallback if no documents matched
                stacked_embs = torch.zeros(0, self.config.memory_dim, device=device)
                
            batch_embeddings.append(stacked_embs)
            
        # Pad batch if retrieved count differs across inputs
        max_k = max(emb.shape[0] for emb in batch_embeddings)
        
        if max_k > 0:
            padded_embeddings = torch.zeros(B_sz, max_k, self.config.memory_dim, device=device)
            for b in range(B_sz):
                k_size = batch_embeddings[b].shape[0]
                if k_size > 0:
                    padded_embeddings[b, :k_size, :] = batch_embeddings[b]
            memory_embeddings = padded_embeddings
        else:
            # Empty tensor fallback: (B, 0, D_memory)
            memory_embeddings = torch.zeros(B_sz, 0, self.config.memory_dim, device=device)
            
        # Pass through linear projection layers
        if memory_embeddings.shape[1] > 0:
            memory_embeddings = self.output_proj(memory_embeddings)
            
        return memory_embeddings, batch_docs
