import time
import queue
import logging
import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDDEVICE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vanitas.audio.capture")

class AudioCapture:
    """Captures streaming microphone input with fallback to synthetic audio for headless runs."""
    
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 512, channels: int = 1, dtype: str = "float32"):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.dtype = dtype
        
        self.queue = queue.Queue()
        self.stream = None
        self.is_running = False
        
        # Check if actual sound card exists
        self.has_physical_device = False
        if SOUNDDEVICE_AVAILABLE:
            try:
                devices = sd.query_devices()
                default_input = sd.query_devices(kind='input')
                if default_input is not None:
                    self.has_physical_device = True
            except Exception as e:
                logger.warning(f"Error querying audio devices: {e}. Falling back to synthetic audio generator.")
        else:
            logger.warning("sounddevice library not fully loaded (or missing backend). Falling back to synthetic audio generator.")

    def _audio_callback(self, indata, frames, time_info, status):
        """Internal callback for sounddevice InputStream."""
        if status:
            logger.warning(f"Audio capture warning: {status}")
        self.queue.put(indata.copy())

    def start(self):
        """Start capturing audio."""
        if self.is_running:
            return
            
        self.is_running = True
        # Clear out any old elements in queue
        while not self.queue.empty():
            self.queue.get_nowait()
            
        if self.has_physical_device:
            logger.info("Initializing physical microphone capture stream via sounddevice...")
            try:
                self.stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.chunk_size,
                    device=None, # Use default system input
                    channels=self.channels,
                    dtype=self.dtype,
                    callback=self._audio_callback
                )
                self.stream.start()
                logger.info("Physical audio capture started.")
            except Exception as e:
                logger.error(f"Failed to start physical sounddevice capture: {e}. Switching to synthetic stream.")
                self.has_physical_device = False
        
        if not self.has_physical_device:
            logger.info("Starting synthetic voice audio generator (headless fallback mode)...")

    def stop(self):
        """Stop capturing audio."""
        if not self.is_running:
            return
            
        self.is_running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error closing audio input stream: {e}")
            self.stream = None
        logger.info("Audio capture stopped.")

    def read_generator(self):
        """Generator yielding float32 PCM numpy arrays of size (chunk_size, channels)."""
        if not self.is_running:
            self.start()
            
        t = 0.0
        while self.is_running:
            if self.has_physical_device:
                try:
                    # Non-blocking get with short timeout to keep loop responsive
                    chunk = self.queue.get(timeout=0.1)
                    yield chunk
                except queue.Empty:
                    continue
            else:
                # Generate synthetic mock voice data (sine waves overlayed with silence intervals)
                # This simulates natural talking bursts: 2.0s talking, 2.5s silence
                time.sleep(self.chunk_size / self.sample_rate)
                cycle_time = 4.5
                mod_time = t % cycle_time
                
                # Active speech window: 0s to 2.0s
                if mod_time < 2.0:
                    # Simulating a speech wave (frequencies 120Hz + 240Hz + 600Hz harmonics)
                    samples = np.arange(self.chunk_size)
                    rad_s = 2 * np.pi / self.sample_rate
                    wave = (
                        0.4 * np.sin(150 * rad_s * (samples + t * self.sample_rate)) + 
                        0.2 * np.sin(300 * rad_s * (samples + t * self.sample_rate)) + 
                        0.15 * np.sin(750 * rad_s * (samples + t * self.sample_rate))
                    )
                    # Add tiny noise
                    noise = np.random.normal(0, 0.01, self.chunk_size)
                    chunk = (wave + noise).astype(np.float32)
                else:
                    # Silence / background room tone
                    chunk = np.random.normal(0, 0.002, self.chunk_size).astype(np.float32)
                
                t += self.chunk_size / self.sample_rate
                # Ensure correct shape (chunk_size, channels)
                yield chunk.reshape(-1, self.channels)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
