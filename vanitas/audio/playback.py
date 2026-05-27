import time
import queue
import threading
import logging
import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDDEVICE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vanitas.audio.playback")

class AudioPlayback:
    """Plays streaming output audio chunks with low latency, interruptibility (barge-in support), and headless fallback."""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1, dtype: str = "float32"):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        
        self.play_queue = queue.Queue()
        self.stream = None
        self.is_running = False
        self.active_playing = False
        self.playback_thread = None
        self.interrupted = False
        
        # Check if actual sound card exists
        self.has_physical_device = False
        if SOUNDDEVICE_AVAILABLE:
            try:
                devices = sd.query_devices()
                default_output = sd.query_devices(kind='output')
                if default_output is not None:
                    self.has_physical_device = True
            except Exception as e:
                logger.warning(f"Error querying audio output devices: {e}. Falling back to virtual playback simulator.")
        else:
            logger.warning("sounddevice library not fully loaded (or missing backend). Falling back to virtual playback simulator.")

    def start(self):
        """Start the background playback processor."""
        if self.is_running:
            return
            
        self.is_running = True
        self.interrupted = False
        # Clear out play queue
        while not self.play_queue.empty():
            self.play_queue.get_nowait()
            
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()
        logger.info("Audio playback engine started.")

    def stop(self):
        """Stop the playback processor."""
        if not self.is_running:
            return
            
        self.is_running = False
        self.interrupt()
        
        if self.playback_thread is not None:
            self.playback_thread.join(timeout=1.0)
            self.playback_thread = None
            
        logger.info("Audio playback engine stopped.")

    def play(self, audio_data: np.ndarray):
        """Enqueue an audio numpy array (mono/stereo) for playback.
        
        Args:
            audio_data: Numpy array of shape (N,) or (N, channels) containing PCM samples.
        """
        if not self.is_running:
            self.start()
            
        # Ensure correct shape
        if len(audio_data.shape) == 1:
            audio_data = audio_data.reshape(-1, self.channels)
            
        # Push to queue in chunks (smaller chunks enable faster interruption response)
        chunk_size = 2048
        n_samples = len(audio_data)
        for i in range(0, n_samples, chunk_size):
            chunk = audio_data[i:i + chunk_size]
            self.play_queue.put(chunk)

    def interrupt(self):
        """Instantly interrupt any current playback and clear all enqueued audio (Barge-In)."""
        self.interrupted = True
        
        # Clear the queue
        while not self.play_queue.empty():
            try:
                self.play_queue.get_nowait()
            except queue.Empty:
                break
                
        # Force stop the sounddevice stream if active
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
            
        self.active_playing = False
        logger.info("Playback interrupted! Playout queue flushed (Barge-In).")
        self.interrupted = False  # Reset for next run

    def _playback_loop(self):
        """Runs in background thread to pull audio chunks and stream them to speaker."""
        while self.is_running:
            try:
                # Timeout allows thread to check is_running flag regularly
                chunk = self.play_queue.get(timeout=0.1)
            except queue.Empty:
                self.active_playing = False
                continue
                
            self.active_playing = True
            
            if self.has_physical_device:
                try:
                    # Initialize active stream on-demand to support dynamic stop/close operations
                    self.stream = sd.OutputStream(
                        samplerate=self.sample_rate,
                        channels=self.channels,
                        dtype=self.dtype
                    )
                    self.stream.start()
                    
                    # Write block synchronously (since it's a dedicated background thread)
                    self.stream.write(chunk)
                    
                    if self.stream is not None:
                        self.stream.stop()
                        self.stream.close()
                        self.stream = None
                except Exception as e:
                    logger.error(f"Sounddevice playback error: {e}. Switching to virtual playback.")
                    self.has_physical_device = False
            
            # Virtual playback fallback: sleep to simulate physical playback duration
            if not self.has_physical_device and not self.interrupted:
                duration = len(chunk) / self.sample_rate
                # Sleep in small slices to remain interruptible
                slice_time = 0.05
                elapsed = 0.0
                while elapsed < duration and not self.interrupted:
                    sleep_len = min(slice_time, duration - elapsed)
                    time.sleep(sleep_len)
                    elapsed += sleep_len

        self.active_playing = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
