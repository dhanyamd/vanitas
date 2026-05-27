import os
import sys
import torch
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vanitas.model.config import VanitasModelConfig
from vanitas.model.vanitas import VanitasModel

def test_perception_flow():
    print("="*60)
    print("⚡ VANITAS PERCEPTION STREAM & FUSION GATES INTEGRATION TEST ⚡")
    print("="*60)
    
    # 1. Config
    print("[Step 1] Initializing model parameters...")
    config = VanitasModelConfig(
        perception_layers=4,    # Test with 4 blocks for speed
        perception_dim=256,     # D_model = 256
        gate_hidden_dim=128
    )
    
    # 2. Instantiate Model
    print("[Step 2] Building unified VanitasModel...")
    model = VanitasModel(config)
    
    # Log total parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model successfully built. Active parameters: {total_params:,} (~{total_params/1e6:.2f}M)")
    
    # 3. Simulate continuous audio log-mel features
    # Batch size = 2, Time sequence length = 50 frames (500ms of audio), Mel bins = 80
    batch_size = 2
    seq_len = 50
    print(f"\n[Step 3] Simulating continuous mel-spectrogram tensor input of shape (B={batch_size}, T={seq_len}, Mel={config.mel_bins})...")
    mel_input = torch.randn(batch_size, seq_len, config.mel_bins)
    
    print("\n[Step 4] Executing forward pass...")
    # Forward pass
    outputs = model(mel_input)
    perception_outputs = outputs["perception_outputs"]
    perception_state = outputs["perception_state"]
    think_gate = outputs["think_gate"]
    backchannel_gate = outputs["backchannel_gate"]
    speak_gate = outputs["speak_gate"]
    
    print("\n🟢 Model Forward Completed. Output Dimension Verification:")
    print(f"Perception Stream Sequence Outputs: {list(perception_outputs.shape)} (Expected: [2, 50, 256])")
    print(f"Perception State Final Snapshots:    {list(perception_state.shape)}  (Expected: [2, 256])")
    print(f"Think Gate Gating Activations (P->C):  {list(think_gate.shape)}   (Expected: [2, 50, 1])")
    print(f"Backchannel Gate Bypass (P->G):      {list(backchannel_gate.shape)}   (Expected: [2, 50, 1])")
    print(f"Speak Gate Release Signal (C->G):    {list(speak_gate.shape)}   (Expected: [2, 50, 1])")
    
    # Assertions
    assert perception_outputs.shape == (batch_size, seq_len, config.perception_dim)
    assert perception_state.shape == (batch_size, config.perception_dim)
    assert think_gate.shape == (batch_size, seq_len, 1)
    
    # Verify Sigmoid range
    assert torch.all(think_gate >= 0.0) and torch.all(think_gate <= 1.0), "Think gate values must lie in Sigmoid range [0, 1]"
    assert torch.all(backchannel_gate >= 0.0) and torch.all(backchannel_gate <= 1.0), "Backchannel gate values must lie in Sigmoid range [0, 1]"
    
    print("\n🏆 Assertions passed! Gating output values successfully constrained within Sigmoid boundaries.")
    print("="*60)
    print("🎉 Integration test completed successfully! 🎉")
    print("="*60)

if __name__ == "__main__":
    test_perception_flow()
