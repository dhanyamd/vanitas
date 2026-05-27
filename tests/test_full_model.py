import torch
from vanitas.model.vanitas import VanitasModel
from vanitas.model.config import VanitasModelConfig

def test_vanitas_model_forward():
    # Use smaller config for testing locally
    config = VanitasModelConfig(
        perception_dim=128,
        cognition_dim=128,
        production_dim=128,
        memory_dim=128,
        mel_bins=80,
        perception_layers=2,
        cognition_layers=2,
        production_layers=2,
        cognition_heads=2
    )
    
    model = VanitasModel(config)
    
    batch_size = 2
    seq_len = 50
    
    # Dummy inputs
    mel_frames = torch.randn(batch_size, seq_len, config.mel_bins)
    memory_embeddings = torch.randn(batch_size, 5, config.memory_dim)
    
    # 1. Test inference (no time_steps)
    model.eval()
    with torch.no_grad():
        audio, v_pred, gates = model(mel_frames, memory_embeddings)
        
        assert audio is not None, "Inference should return audio waveforms"
        assert v_pred is None, "Inference shouldn't return velocity predictions"
        
        # Audio length = seq_len * hop_length
        assert audio.shape == (batch_size, 1, seq_len * config.hop_length)
        
        think, backchannel, speak = gates
        assert think.shape == (batch_size, seq_len, 1)
        
    # 2. Test training (with time_steps)
    model.train()
    time_steps = torch.rand(batch_size, 1)
    
    audio, v_pred, gates = model(mel_frames, memory_embeddings, time_steps)
    
    assert audio is None, "Training shouldn't return audio waveforms"
    assert v_pred is not None, "Training should return velocity predictions"
    assert v_pred.shape == (batch_size, seq_len, config.mel_bins)
    
    print("Full Vanitas Architecture (3-Stream) forward pass successful!")

if __name__ == "__main__":
    test_vanitas_model_forward()
