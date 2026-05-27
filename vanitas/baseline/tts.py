import os
import time
import torch
import numpy as np
import logging

try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vanitas.baseline.tts")

class BaselineTTS:
    """Wraps Kokoro TTS for low-latency local speech synthesis with a fully operational mock fallback."""
    
    def __init__(self, voice: str = "af_heart", speed: float = 1.0, device: str = "auto"):
        self.voice = voice
        self.speed = speed
        self.device = device
        self.pipeline = None
        
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
                
        if KOKORO_AVAILABLE:
            try:
                logger.info(f"Initializing Kokoro TTS KPipeline on device '{self.device}'...")
                # Kokoro uses lang_code='a' for American English, 'b' for British
                self.pipeline = KPipeline(lang_code='a', device=self.device)
                logger.info("Kokoro TTS initialized successfully.")
            except Exception as e:
                logger.warning(
                    f"Failed to load Kokoro TTS pipeline ({e}). "
                    "Falling back to Mock Audio Synthesizer."
                )
                self.pipeline = None
        else:
            logger.warning("kokoro library is not installed. Falling back to Mock Audio Synthesizer.")

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesizes text into a float32 PCM numpy array (16kHz, mono).
        
        Args:
            text: The response string to synthesize.
            
        Returns:
            np.ndarray: Audio data in float32 format, sampled at 16000Hz.
        """
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
            
        logger.info(f"Synthesizing text: '{text[:50]}...'")
        
        if self.pipeline is not None:
            try:
                # pipeline yields (graphemes, phonemes, audio_tensor)
                generator = self.pipeline(
                    text,
                    voice=self.voice,
                    speed=self.speed,
                    split_pattern=r'\n+' # Keep sentences unified for natural prosody
                )
                
                audio_segments = []
                for _, _, audio in generator:
                    if audio is not None:
                        # Kokoro outputs float32 numpy arrays natively
                        audio_segments.append(audio)
                        
                if len(audio_segments) == 0:
                    return np.zeros(0, dtype=np.float32)
                    
                combined_audio = np.concatenate(audio_segments)
                logger.info(f"Kokoro Synthesis Complete. Generated {len(combined_audio)/16000:.2f}s of audio.")
                return combined_audio
            except Exception as e:
                logger.error(f"Error during Kokoro synthesis: {e}. Falling back to mock synthesis.")
                return self._mock_synthesize(text)
        else:
            return self._mock_synthesize(text)

    def _mock_synthesize(self, text: str) -> np.ndarray:
        """Mock TTS synthesizer that creates modulated synthetic vocal-like audio chunks."""
        words = text.split()
        num_words = len(words)
        
        # Human talking rate: approx 150 words per minute -> 2.5 words per second
        words_per_sec = 2.5
        duration_sec = max(0.5, num_words / words_per_sec)
        
        logger.info(f"Mock TTS Generating virtual speech (Duration: {duration_sec:.2f}s, Words: {num_words})")
        
        sample_rate = 16000
        total_samples = int(duration_sec * sample_rate)
        
        # Synthesize a speech-like envelope with a fundamental frequency (150Hz) modulated by a low-frequency wave (2Hz)
        t = np.arange(total_samples)
        rad_s = 2 * np.pi / sample_rate
        
        # Fundamental frequency + harmonics
        voice_carrier = (
            0.5 * np.sin(160 * rad_s * t) + 
            0.25 * np.sin(320 * rad_s * t) + 
            0.15 * np.sin(480 * rad_s * t)
        )
        
        # Syllabic envelope modulation (simulate syllables using 3.5Hz sine wave)
        syllables_envelope = 0.5 * (1.0 + np.sin(3.5 * 2 * np.pi * t / sample_rate))
        
        # Sentence envelope (fade in and out at boundaries)
        sentence_envelope = np.ones_like(t, dtype=np.float32)
        fade_len = int(0.15 * sample_rate)  # 150ms fade
        if total_samples > 2 * fade_len:
            sentence_envelope[:fade_len] = np.linspace(0.0, 1.0, fade_len)
            sentence_envelope[-fade_len:] = np.linspace(1.0, 0.0, fade_len)
            
        mock_speech = voice_carrier * syllables_envelope * sentence_envelope
        
        # Add a tiny bit of comfort noise
        comfort_noise = np.random.normal(0, 0.005, total_samples)
        
        return (mock_speech + comfort_noise).astype(np.float32)
