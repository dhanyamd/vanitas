import torch
import torchaudio
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path

def pcm_to_float32(pcm_data: bytes) -> np.ndarray:
    """Converts raw 16-bit PCM bytes to a float32 numpy array in range [-1.0, 1.0]."""
    return np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

def float32_to_pcm16(float_data: np.ndarray) -> np.ndarray:
    """Converts a float32 numpy array in range [-1.0, 1.0] back to 16-bit PCM numpy array."""
    clamped = np.clip(float_data, -1.0, 1.0)
    return (clamped * 32767.0).astype(np.int16)

def resample_audio(audio_tensor: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resamples a PyTorch audio tensor from orig_sr to target_sr using torchaudio resampler."""
    if orig_sr == target_sr:
        return audio_tensor
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    return resampler(audio_tensor)

def save_wav(filepath: str | Path, audio_data: np.ndarray, sample_rate: int = 16000):
    """Saves a float32 or int16 numpy array as a standard WAV file.
    
    Args:
        filepath: Output WAV file path.
        audio_data: Numpy array of shape (N,) or (N, channels).
        sample_rate: The audio sample rate.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # If float32, write_to_wav will accept it directly or we convert to int16 for compatibility
    if audio_data.dtype == np.float32:
        # Scale to int16 to ensure high device compatibility across media players
        wav_data = float32_to_pcm16(audio_data)
    else:
        wav_data = audio_data
        
    wav.write(str(path), sample_rate, wav_data)
