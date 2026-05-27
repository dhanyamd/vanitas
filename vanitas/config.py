import os
from dataclasses import dataclass, field
from pathlib import Path

# Base Paths
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINTS_DIR = WORKSPACE_ROOT / "checkpoints"

# Create directories if they don't exist
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

@dataclass
class AudioConfig:
    """Settings for real-time audio capturing and feature extraction."""
    sample_rate: int = 16000          # Standard for voice ML
    channels: int = 1                 # Mono for capturing / single stream
    chunk_size: int = 512             # 32ms window at 16kHz (low latency processing block)
    dtype: str = "float32"            # PyTorch standard
    
    # Mel Spectrogram parameters (standard speech settings)
    n_fft: int = 400                  # 25ms window size
    hop_length: int = 160             # 10ms step size (100 frames/sec)
    win_length: int = 400
    n_mels: int = 80                  # 80-dimensional continuous representations
    f_min: float = 0.0
    f_max: float = 8000.0             # Nyquist limit for 16kHz audio

@dataclass
class BaselineVADConfig:
    """Settings for baseline Voice Activity Detection using Silero VAD."""
    threshold: float = 0.5            # Speech probability threshold
    min_speech_duration_ms: int = 250 # Ignore shorter sounds
    min_silence_duration_ms: int = 400 # Wait this long before endpointing in baseline
    sample_rate: int = 16000

@dataclass
class BaselineASRConfig:
    """Settings for baseline speech-to-text with faster-whisper."""
    model_size: str = "distil-large-v3" # Fast & highly accurate distilled large Whisper
    device: str = "auto"              # Will fallback to CPU/MPS/CUDA automatically
    compute_type: str = "default"     # int8, float16 or float32

@dataclass
class BaselineTTSConfig:
    """Settings for baseline speech synthesis with Kokoro."""
    model_name: str = "kokoro"
    voice: str = "af_heart"           # Default warm American English female voice
    speed: float = 1.0
    device: str = "auto"

@dataclass
class GlobalConfig:
    """Consolidated application configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: BaselineVADConfig = field(default_factory=BaselineVADConfig)
    asr: BaselineASRConfig = field(default_factory=BaselineASRConfig)
    tts: BaselineTTSConfig = field(default_factory=BaselineTTSConfig)
    
    # Paths
    workspace_root: Path = WORKSPACE_ROOT
    data_dir: Path = DATA_DIR
    checkpoints_dir: Path = CHECKPOINTS_DIR
    
    # Dev settings
    verbose: bool = True
