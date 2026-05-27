import torch
import torch.nn as nn
import numpy as np
import logging

try:
    import torchaudio.transforms as T
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    try:
        import librosa
        LIBROSA_AVAILABLE = True
    except ImportError:
        LIBROSA_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vanitas.audio.features")

def _compute_numpy_mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Computes standard triangular Mel-filterbank matrix in pure NumPy."""
    # Convert Hz to Mel
    f_min = 0.0
    f_max = sr / 2.0
    mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
    
    # Linear spacing in Mel scale
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    # Convert Mel points back to Hz
    hz_points = 700.0 * (10.0**(mel_points / 2595.0) - 1.0)
    
    # Map Hz frequencies to FFT bins
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    # Construct filterbank matrix
    fbank = torch.zeros(n_mels, n_fft // 2 + 1, dtype=torch.float32)
    for m in range(1, n_mels + 1):
        # Left slope
        for k in range(bins[m - 1], bins[m]):
            fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        # Right slope
        for k in range(bins[m], bins[m + 1]):
            fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
            
    return fbank


class MelExtractor:
    """Computes streaming log-mel spectrogram features with zero-latency buffer alignment."""
    
    def __init__(self, sample_rate: int = 16000, n_fft: int = 400, hop_length: int = 160, win_length: int = 400, n_mels: int = 80):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        
        self.mel_transform = None
        self.fbank = None
        self.window = None
        
        if TORCHAUDIO_AVAILABLE:
            self.mel_transform = T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                power=2.0,
                center=False
            )
        else:
            logger.info("Torchaudio not available. Loading triangular Mel-filterbanks in pure NumPy/PyTorch fallback.")
            self.fbank = _compute_numpy_mel_filterbank(self.sample_rate, self.n_fft, self.n_mels)
            # Create a Hann window tensor
            self.window = torch.from_numpy(np.hanning(self.win_length)).float()
            
        # Internal audio buffer to hold samples that have not yet formed a complete hop window
        self.audio_buffer = torch.zeros(0, dtype=torch.float32)

    def reset(self):
        """Clears the streaming buffer."""
        self.audio_buffer = torch.zeros(0, dtype=torch.float32)

    def feed_audio(self, pcm_data: np.ndarray) -> torch.Tensor:
        """Appends new PCM samples and extracts any complete mel spectrogram frames.
        
        Args:
            pcm_data: Numpy array of shape (N,) or (N, 1) containing raw PCM audio.
            
        Returns:
            torch.Tensor: Log-mel frames of shape (1, T_new, n_mels) where T_new is the number
                          of complete 10ms hop frames extracted. Can have T_new = 0.
        """
        # Ensure mono float32 tensor
        if isinstance(pcm_data, np.ndarray):
            pcm_tensor = torch.from_numpy(pcm_data).float()
        else:
            pcm_tensor = pcm_data.float()
            
        if len(pcm_tensor.shape) > 1:
            pcm_tensor = pcm_tensor.squeeze(-1) # Convert (N, 1) to (N,)
            
        # Append to our circular rolling buffer
        self.audio_buffer = torch.cat([self.audio_buffer, pcm_tensor])
        
        # We need at least win_length samples to compute the first window.
        # Each subsequent frame requires hop_length more samples.
        if len(self.audio_buffer) < self.win_length:
            return torch.zeros(1, 0, self.n_mels)
            
        # Calculate how many complete frames we can compute
        available_samples = len(self.audio_buffer)
        num_frames = (available_samples - self.win_length) // self.hop_length + 1
        
        if num_frames <= 0:
            return torch.zeros(1, 0, self.n_mels)
            
        # Extract the exact range of samples needed
        samples_to_process = self.win_length + (num_frames - 1) * self.hop_length
        pcm_chunk = self.audio_buffer[:samples_to_process]
        
        # Update the buffer (retian the remaining samples at the end for overlap)
        self.audio_buffer = self.audio_buffer[samples_to_process - (self.win_length - self.hop_length):]
        
        if TORCHAUDIO_AVAILABLE:
            # Compute mel spectrogram using torchaudio
            mel_spec = self.mel_transform(pcm_chunk.unsqueeze(0))
        elif LIBROSA_AVAILABLE:
            # Compute mel spectrogram using librosa
            S = librosa.feature.melspectrogram(
                y=pcm_chunk.numpy(),
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
                power=2.0,
                center=False
            )
            mel_spec = torch.from_numpy(S).unsqueeze(0)
        else:
            # Compute mel spectrogram using our custom pure NumPy/PyTorch DSP scan
            # We slice the chunk into overlapping frames
            frames = []
            for t in range(num_frames):
                start = t * self.hop_length
                frame = pcm_chunk[start:start + self.win_length]
                # Apply Hann window
                windowed = frame * self.window
                # Compute Real FFT magnitude squared
                fft_out = torch.fft.rfft(windowed, n=self.n_fft)
                power_spectrum = torch.abs(fft_out).pow(2.0)
                frames.append(power_spectrum)
                
            # Stack frames along column dimension: (num_frames, n_fft // 2 + 1)
            stacked_frames = torch.stack(frames, dim=1) # (n_fft // 2 + 1, num_frames)
            
            # Apply Mel-filterbanks matrix multiplication: (n_mels, n_fft // 2 + 1) x (n_fft // 2 + 1, num_frames)
            mel_spec = torch.mm(self.fbank, stacked_frames).unsqueeze(0) # (1, n_mels, num_frames)
        
        # Log scale: log10(mel_spec + epsilon)
        log_mel = torch.log10(torch.clamp(mel_spec, min=1e-5))
        
        # Permute to standard model format: (Batch, Time, Features) -> (1, num_frames, n_mels)
        return log_mel.permute(0, 2, 1)


class MelProjection(nn.Module):
    """Learned neural layer that projects raw 80-dim continuous mel features into the model space."""
    
    def __init__(self, n_mels: int = 80, model_dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_mels, model_dim),
            nn.LayerNorm(model_dim),
            nn.SiLU()
        )
        
    def forward(self, mel_frames: torch.Tensor) -> torch.Tensor:
        return self.proj(mel_frames)
