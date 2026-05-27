import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vanitas.model.config import VanitasModelConfig
from vanitas.model.cognition.core import CognitionCore
from vanitas.model.cognition.memory import VectorMemory
from vanitas.model.cognition.retrieval import FactualRetriever

def test_vector_memory():
    print("\n[Test 1] Validating Vector Memory (DB layer)...")
    memory = VectorMemory(embedding_dim=512, collection_name="test_collection")
    
    # Generate mock facts
    facts = [
        "Mamba-2 is a Structured State Space Duality model with O(1) step inference.",
        "Compute-Memory Separation isolates fact storage from reasoning weights.",
        "The Vanitas architecture uses three parallel neural streams."
    ]
    
    # Generate distinct mock embeddings (simulating distinct semantic points)
    rng = np.random.default_rng(42)
    embs = [rng.normal(0.0, 0.1, 512).astype(np.float32) for _ in range(3)]
    
    # Add to database
    for idx, (fact, emb) in enumerate(zip(facts, embs)):
        memory.add_fact(fact, emb, f"fact_{idx}")
        
    print("🟢 Facts added to vector memory successfully.")
    
    # Query with a vector close to the second fact (emb[1])
    query_vector = embs[1] + rng.normal(0.0, 0.01, 512).astype(np.float32) # Add small noise
    results = memory.query_semantic(query_vector, top_k=1)
    
    assert len(results) == 1, "Should retrieve exactly 1 document"
    assert "Separation" in results[0]["document"], "Should retrieve closest semantic fact (Fact 1)"
    print(f"🟢 Semantic Retrieval matches! Found document: '{results[0]['document'][:50]}...'")


def test_factual_retriever():
    print("\n[Test 2] Validating Factual Retriever (PyTorch interface layer)...")
    config = VanitasModelConfig(memory_dim=512)
    retriever = FactualRetriever(config)
    
    # Inject dummy facts into retriever's internal database
    rng = np.random.default_rng(100)
    retriever.memory.add_fact(
        "Standard VAD suffers from static silence delay thresholds.",
        rng.normal(0.0, 0.1, 512).astype(np.float32),
        "vad_fact"
    )
    
    # Perform batched query: Batch size = 2, Query dim = 512
    # Create query state tensor
    query_tensor = torch.from_numpy(rng.normal(0.0, 0.1, (2, 512)).astype(np.float32))
    
    # Forward pass through retriever
    memory_embeddings, matched_docs = retriever(query_tensor, top_k=2)
    
    print(f"Retrieved embeddings shape: {list(memory_embeddings.shape)}")
    print(f"Retrieved documents: {matched_docs}")
    
    # Assertions
    assert memory_embeddings.shape[0] == 2, "Batch dimension must match"
    assert len(matched_docs) == 2, "Must return document matches per batch item"
    assert memory_embeddings.shape[2] == 512, "Feature dimensions must align with memory_dim"
    print("🟢 Factual Retriever output shapes and batched routing validated.")


def test_cognition_core_reasoning():
    print("\n[Test 3] Validating Cognition Core (Transformer layer)...")
    config = VanitasModelConfig(
        perception_dim=512,
        cognition_dim=512,
        memory_dim=512,
        cognition_layers=2, # Small layers for test
        cognition_heads=4
    )
    
    core = CognitionCore(config)
    
    # Simulate batched input: Batch = 2, perception_dim = 512
    perception_state = torch.randn(2, 512)
    # Simulate batched retrieved facts: Batch = 2, K = 3 documents, memory_dim = 512
    memory_embeddings = torch.randn(2, 3, 512)
    
    # Forward through reasoning core
    cognition_state = core(perception_state, memory_embeddings)
    
    print(f"Cognition state output shape: {list(cognition_state.shape)}")
    
    # Assertions
    assert cognition_state.shape == (2, 512), "Output state shape must match (B, cognition_dim)"
    print("🟢 Cognition Core Multi-Head Cross-Attention reasoning validated.")


def test_end_to_end_cognition_pipeline():
    print("\n[Test 4] Validating End-to-End Cognition & RAG Pipeline...")
    config = VanitasModelConfig(
        perception_dim=512,
        cognition_dim=512,
        memory_dim=512,
        cognition_layers=2,
        cognition_heads=4
    )
    
    # 1. Initialize components
    memory = VectorMemory(embedding_dim=512, collection_name="e2e_collection")
    retriever = FactualRetriever(config, memory)
    core = CognitionCore(config)
    
    # 2. Add some factual guidelines
    rng = np.random.default_rng(200)
    memory.add_fact(
        "Flow Matching operates as a continuous generative head, removing discrete tokenizers.",
        rng.normal(0.0, 0.1, 512).astype(np.float32),
        "flow_fact"
    )
    
    # 3. Simulate incoming Perception State from Mamba-2 Perception Stream
    perception_state = torch.from_numpy(rng.normal(0.0, 0.1, (2, 512)).astype(np.float32))
    
    # 4. Step A: Query memory and retrieve dense factual embeddings
    memory_embeddings, matched_docs = retriever(perception_state, top_k=1)
    
    # 5. Step B: Feed both state vectors to the Cognition Core for reasoning
    cognition_state = core(perception_state, memory_embeddings)
    
    print(f"Final output Cognition State shape: {list(cognition_state.shape)}")
    
    # Assertions
    assert cognition_state.shape == (2, 512), "Final output shape must be (B, D_cognition)"
    print("🟢 End-to-End RAG Perception-to-Cognition pipeline successfully integrated.")


def run_all_tests():
    print("="*60)
    print("🧪 VANITAS COGNITION & EXTERNAL MEMORY UNIT TESTS 🧪")
    print("="*60)
    
    test_vector_memory()
    test_factual_retriever()
    test_cognition_core_reasoning()
    test_end_to_end_cognition_pipeline()
    
    print("\n" + "="*60)
    print("🎉 ALL COGNITION TESTS PASSED SUCCESSFULLY! 🎉")
    print("="*60)

if __name__ == "__main__":
    run_all_tests()
