import os
import torch
import numpy as np
import logging

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vanitas.baseline.asr")

class BaselineASR:
    """Wraps faster-whisper for low-latency local speech-to-text transcription with mock fallback."""
    
    def __init__(self, model_size: str = "distil-large-v3", device: str = "auto", compute_type: str = "default"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        
        # Determine device
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                # Float16 is standard and highly accelerated on NVIDIA GPUs
                if self.compute_type == "default":
                    self.compute_type = "float16"
            else:
                self.device = "cpu"
                # Int8 is extremely fast on CPU
                if self.compute_type == "default":
                    self.compute_type = "int8"
                    
        if FASTER_WHISPER_AVAILABLE:
            try:
                logger.info(f"Loading faster-whisper model '{self.model_size}' on '{self.device}' ({self.compute_type})...")
                # Suppress download logs to stdout
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=None # Use default cache dir
                )
                logger.info("faster-whisper model loaded successfully.")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize faster-whisper model ({e}). "
                    "Falling back to Mock ASR."
                )
                self.model = None
        else:
            logger.warning("faster-whisper library is not installed. Falling back to Mock ASR.")

    def transcribe(self, audio_data: np.ndarray, language: str = "en") -> str:
        """Transcribes the given float32/int16 PCM audio array to text.
        
        Args:
            audio_data: Numpy array of shape (N,) containing raw PCM audio.
            language: Target ISO-639-1 language code.
            
        Returns:
            str: The transcribed text response.
        """
        # Ensure flat shape
        if len(audio_data.shape) > 1:
            audio_data = audio_data.squeeze(-1)
            
        if self.model is not None:
            try:
                # transcribes takes a numpy array in float32 directly
                # It requires values to be normalized in [-1, 1]
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                    
                segments, info = self.model.transcribe(
                    audio_data,
                    beam_size=5,
                    language=language,
                    vad_filter=True, # Secondary VAD check for cleaner text
                    vad_parameters=dict(min_silence_duration_ms=400)
                )
                
                # Consume segments generator
                transcript_segments = [seg.text for seg in segments]
                transcript = "".join(transcript_segments).strip()
                logger.info(f"ASR Transcript: '{transcript}' (Confidence: {info.language_probability:.2f})")
                return transcript
            except Exception as e:
                logger.error(f"Error during faster-whisper transcription: {e}. Falling back to mock output.")
                return self._mock_transcribe(audio_data)
        else:
            return self._mock_transcribe(audio_data)

    def _mock_transcribe(self, audio_data: np.ndarray) -> str:
        """Generates a high-quality mock transcription based on audio energy and duration."""
        duration_sec = len(audio_data) / 16000.0
        
        # Calculate overall energy
        energy = np.sqrt(np.mean(audio_data**2))
        
        # If practically silent
        if energy < 0.005:
            return ""
            
        logger.info(f"Mock ASR Triggered (Duration: {duration_sec:.2f}s, Energy: {energy:.4f})")
        
        # Return generic conversation responses corresponding to duration
        if duration_sec < 1.0:
            return "Yes."
        elif duration_sec < 2.5:
            return "Hello, can you hear me?"
        elif duration_sec < 4.5:
            return "Could you please explain how the new neural fusion gate works in the paper?"
        else:
            return "That sounds incredibly fascinating! Let's do a deep-dive research into how we can train our smaller model with infinite memory separation."
