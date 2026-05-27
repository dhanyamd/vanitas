import os
import numpy as np
import logging
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from vanitas.config import GlobalConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vanitas.model.cognition.memory")


class MockVectorDB:
    """A pure NumPy-native Vector Database implementing exact cosine similarity.
    Acts as a lightweight, internet-resilient fallback for offline testing.
    """
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []

    def add(self, documents: list[str], embeddings: list[np.ndarray], ids: list[str], metadatas: list[dict] = None):
        for idx, (doc, emb, d_id) in enumerate(zip(documents, embeddings, ids)):
            self.documents.append(doc)
            # Ensure correct shape
            emb = np.array(emb, dtype=np.float32).flatten()
            if emb.shape[0] != self.embedding_dim:
                # Pad or truncate to match dimensions
                padded = np.zeros(self.embedding_dim, dtype=np.float32)
                size = min(self.embedding_dim, emb.shape[0])
                padded[:size] = emb[:size]
                emb = padded
                
            self.embeddings.append(emb)
            self.ids.append(d_id)
            self.metadatas.append(metadatas[idx] if metadatas else {})

    def query(self, query_embeddings: np.ndarray, top_k: int = 3) -> dict:
        """Finds closest documents using cosine similarity."""
        if not self.embeddings:
            return {"documents": [[]], "ids": [[]], "metadatas": [[]], "distances": [[]]}
            
        q_emb = np.array(query_embeddings, dtype=np.float32).flatten()
        if q_emb.shape[0] != self.embedding_dim:
            padded = np.zeros(self.embedding_dim, dtype=np.float32)
            size = min(self.embedding_dim, q_emb.shape[0])
            padded[:size] = q_emb[:size]
            q_emb = padded
            
        # Stack all database embeddings: (N, D)
        db_matrix = np.stack(self.embeddings, axis=0)
        
        # Calculate cosine similarity: dot(A, B) / (norm(A) * norm(B))
        q_norm = np.linalg.norm(q_emb)
        db_norms = np.linalg.norm(db_matrix, axis=1)
        
        # Avoid division by zero
        q_norm = 1.0 if q_norm == 0 else q_norm
        db_norms[db_norms == 0] = 1.0
        
        dot_product = np.dot(db_matrix, q_emb)
        cosine_sims = dot_product / (q_norm * db_norms)
        
        # Map similarity to distance (1 - cosine_similarity)
        distances = 1.0 - cosine_sims
        
        # Get top-k sorted indices
        sorted_indices = np.argsort(distances)[:top_k]
        
        ret_docs = [self.documents[i] for i in sorted_indices]
        ret_ids = [self.ids[i] for i in sorted_indices]
        ret_meta = [self.metadatas[i] for i in sorted_indices]
        ret_dist = [float(distances[i]) for i in sorted_indices]
        
        return {
            "documents": [ret_docs],
            "ids": [ret_ids],
            "metadatas": [ret_meta],
            "distances": [ret_dist]
        }


class VectorMemory:
    """🟠 Stream 2 Memory (ChromaDB Wrapper)
    Manages storing and querying dialog facts to achieve Compute-Memory Separation.
    """
    
    def __init__(self, config: GlobalConfig = None, embedding_dim: int = 512, collection_name: str = "vanitas_memory"):
        self.config = config if config is not None else GlobalConfig()
        self.embedding_dim = embedding_dim
        self.collection_name = collection_name
        
        self.db = None
        self.client = None
        self.collection = None
        
        # Attempt to load ChromaDB
        if CHROMADB_AVAILABLE:
            try:
                chroma_path = self.config.workspace_root / "data" / "processed" / "chroma"
                chroma_path.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"Initializing persistent ChromaDB client at '{chroma_path}'...")
                
                # Setup persistent client
                self.client = chromadb.PersistentClient(path=str(chroma_path))
                
                # Get or create collection
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"} # Use cosine similarity metric
                )
                logger.info("ChromaDB vector memory collection initialized successfully.")
            except Exception as e:
                logger.warning(f"Failed to load ChromaDB client ({e}). Falling back to NumPy Vector DB.")
                self.db = MockVectorDB(self.embedding_dim)
        else:
            logger.warning("chromadb library not installed. Falling back to NumPy Vector DB.")
            self.db = MockVectorDB(self.embedding_dim)

    def add_fact(self, text: str, embedding: np.ndarray, fact_id: str, metadata: dict = None):
        """Adds a single factual document with its semantic embedding to the database.
        
        Args:
            text: Raw string document/fact.
            embedding: Semantic embedding array of size (embedding_dim,).
            fact_id: Unique string identifier.
            metadata: Optional dictionary of key-value properties.
        """
        # Ensure flat list format
        embeddings_list = [embedding.flatten().tolist()]
        
        if self.collection is not None:
            try:
                self.collection.add(
                    documents=[text],
                    embeddings=embeddings_list,
                    ids=[fact_id],
                    metadatas=[metadata] if metadata else None
                )
                logger.debug(f"Fact '{fact_id}' successfully added to ChromaDB.")
            except Exception as e:
                logger.error(f"Error writing to ChromaDB: {e}. Writing to fallback db.")
                if self.db is None:
                    self.db = MockVectorDB(self.embedding_dim)
                self.db.add([text], [embedding], [fact_id], [metadata] if metadata else None)
        else:
            self.db.add([text], [embedding], [fact_id], [metadata] if metadata else None)

    def query_semantic(self, query_embedding: np.ndarray, top_k: int = 3) -> list[dict]:
        """Queries the vector database using a semantic query vector.
        
        Args:
            query_embedding: Query embedding of shape (embedding_dim,).
            top_k: Number of relevant facts to retrieve.
            
        Returns:
            list[dict]: A list of retrieved documents with keys 'document', 'id', 'distance', 'metadata'.
        """
        # Ensure float flat list
        q_list = query_embedding.flatten().tolist()
        
        if self.collection is not None:
            try:
                results = self.collection.query(
                    query_embeddings=[q_list],
                    n_results=top_k
                )
                
                # Format results cleanly
                formatted = []
                if results and "documents" in results and len(results["documents"][0]) > 0:
                    docs = results["documents"][0]
                    ids = results["ids"][0]
                    metas = results["metadatas"][0]
                    distances = results["distances"][0]
                    
                    for idx in range(len(docs)):
                        formatted.append({
                            "document": docs[idx],
                            "id": ids[idx],
                            "distance": distances[idx],
                            "metadata": metas[idx] if metas else {}
                        })
                return formatted
            except Exception as e:
                logger.error(f"Error querying ChromaDB: {e}. Executing fallback query.")
                if self.db is None:
                    return []
                # Fallback run
                results = self.db.query(query_embedding, top_k)
        else:
            results = self.db.query(query_embedding, top_k)
            
        # Format fallback results
        formatted = []
        docs = results["documents"][0]
        ids = results["ids"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        
        for idx in range(len(docs)):
            formatted.append({
                "document": docs[idx],
                "id": ids[idx],
                "distance": distances[idx],
                "metadata": metas[idx]
            })
        return formatted
