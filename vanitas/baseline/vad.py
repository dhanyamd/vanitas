import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vanitas.baseline.vad")

class BaselineVAD:
    """Wraps Silero VAD with an endpointing state machine and energy-based fallback."""
    
    def __init__(self, threshold: float = 0.5, min_speech_duration_ms: int = 250, min_silence_duration_ms: int = 400, sample_rate: int = 16000):
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.sample_rate = sample_rate
        
        # Internals for state machine
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_frames = 0
        
        # Calculate frame counts (each frame/chunk is 512 samples = 32ms at 16kHz)
        self.chunk_duration_ms = (512 / 16000) * 1000  # 32ms
        self.speech_threshold_frames = int(min_speech_duration_ms / self.chunk_duration_ms)
        self.silence_threshold_frames = int(min_silence_duration_ms / self.chunk_duration_ms)
        
        # Try loading Silero VAD
        self.model = None
        self.utils = None
        try:
            logger.info("Loading Silero VAD model from PyTorch Hub...")
            # Set trust_repo=True to allow model loading
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            logger.info("Silero VAD model loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Failed to load Silero VAD from PyTorch Hub ({e}). "
                "Falling back to high-performance energy-based VAD (RMS)."
            )
            self.model = None
            
        # Fallback RMS properties
        self.rms_threshold = 0.015    # Root-mean-square amplitude threshold for voice detection
        self.ambient_noise_floor = 0.002
        
    def reset(self):
        """Resets the state machine."""
        self.is_speaking = False
        self.speech_frames = 0
        self.silence_frames = 0

    def process_chunk(self, audio_chunk: np.ndarray) -> bool:
        """Processes a 512-sample float32 audio chunk and returns whether the speaker is actively talking.
        
        Args:
            audio_chunk: Numpy array of shape (512, 1) or (512,)
            
        Returns:
            bool: True if speaking, False if silent/paused.
        """
        # Ensure flat shape
        if len(audio_chunk.shape) > 1:
            audio_chunk = audio_chunk.squeeze(-1)
            
        # 1. Get raw voice probability
        if self.model is not None:
            try:
                # Convert chunk to PyTorch tensor
                tensor_chunk = torch.from_numpy(audio_chunk).float()
                # Run Silero VAD forward pass
                with torch.no_grad():
                    prob = self.model(tensor_chunk, self.sample_rate).item()
                is_active_frame = prob >= self.threshold
            except Exception as e:
                # Fallback to RMS on model failure
                is_active_frame = self._rms_is_active(audio_chunk)
        else:
            is_active_frame = self._rms_is_active(audio_chunk)
            
        # 2. State Machine smoothing (Onset and Offset)
        if is_active_frame:
            self.silence_frames = 0
            if not self.is_speaking:
                self.speech_frames += 1
                if self.speech_frames >= self.speech_threshold_frames:
                    self.is_speaking = True
                    logger.debug("VAD State Transition: SILENT -> SPEAKING")
        else:
            self.speech_frames = 0
            if self.is_speaking:
                self.silence_frames += 1
                if self.silence_frames >= self.silence_threshold_frames:
                    self.is_speaking = False
                    logger.debug("VAD State Transition: SPEAKING -> SILENT")
                    
        return self.is_speaking

    def _rms_is_active(self, audio_chunk: np.ndarray) -> bool:
        """Helper to determine voice activity via energy (RMS)."""
        rms = np.sqrt(np.mean(audio_chunk**2))
        
        # Dynamic ambient noise calibration
        if rms < self.ambient_noise_floor:
            self.ambient_noise_floor = 0.95 * self.ambient_noise_floor + 0.05 * rms
            
        # Threshold scales with the ambient noise floor
        dynamic_threshold = max(self.rms_threshold, self.ambient_noise_floor * 3.5)
        return rms > dynamic_threshold
