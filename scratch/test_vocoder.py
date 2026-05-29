import torch
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path
from transformers import SpeechT5HifiGan
from vanitas.audio.features import MelExtractor

def test_vocoder_transform():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Generate a clean synthetic test audio (vowel sound 'ah' at 220Hz)
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # 220Hz sine wave + 440Hz + 880Hz harmonics
    audio = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 880 * t)
    audio = audio.astype(np.float32)
    
    # Save original synthetic audio to compare
    output_dir = Path("/Users/dhanyamd/.gemini/antigravity/brain/12256c2f-6c4c-441f-8642-afb5c6287392/scratch")
    output_dir.mkdir(parents=True, exist_ok=True)
    wavfile.write(output_dir / "original_synth.wav", sample_rate, audio)
    print("Saved original_synth.wav")

    # 2. Extract base-10 Mel features using our MelExtractor
    extractor = MelExtractor(
        sample_rate=sample_rate,
        n_fft=400,
        hop_length=160,
        win_length=400,
        n_mels=80
    )
    mel_base10 = extractor.feed_audio(audio).to(device) # (1, T, 80)
    print(f"Mel base-10 range: min={mel_base10.min().item():.4f}, max={mel_base10.max().item():.4f}")
    
    # 3. Load SpeechT5 vocoder
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    
    # Test different conversions:
    # A: Base-e conversion: log_e = log_10 * ln(10)
    mel_e = mel_base10 * np.log(10.0)
    print(f"Mel base-e range: min={mel_e.min().item():.4f}, max={mel_e.max().item():.4f}")
    
    with torch.no_grad():
        # SpeechT5 expect log-mel in natural log scale
        audio_out_e = vocoder(mel_e).cpu().numpy().flatten()
        wavfile.write(output_dir / "reconstructed_e.wav", sample_rate, audio_out_e)
        print("Saved reconstructed_e.wav")
        
        # B: Let's also try raw base-10 to see how it differs
        audio_out_raw = vocoder(mel_base10).cpu().numpy().flatten()
        wavfile.write(output_dir / "reconstructed_raw.wav", sample_rate, audio_out_raw)
        print("Saved reconstructed_raw.wav")

if __name__ == "__main__":
    test_vocoder_transform()
