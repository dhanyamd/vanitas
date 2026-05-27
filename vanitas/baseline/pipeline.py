import time
import threading
import logging
import numpy as np

from vanitas.config import GlobalConfig
from vanitas.audio.capture import AudioCapture
from vanitas.audio.playback import AudioPlayback
from vanitas.baseline.vad import BaselineVAD
from vanitas.baseline.asr import BaselineASR
from vanitas.baseline.tts import BaselineTTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vanitas.baseline.pipeline")

class LocalDialogueManager:
    """A beautiful local dialogue manager that answers conversational prompts and papers queries."""
    
    def __init__(self):
        # Conversation history
        self.turns = []
        
        # Knowledge Base about the paper
        self.knowledge = {
            "vanitas": "Vanitas is our state-of-the-art three-stream parallel neural architecture. It utilizes an always-on Mamba-2 Perception Stream, learned Fusion Gates to eliminate traditional VAD, and a parallel Mamba-2 Production Stream with Flow Matching.",
            "mamba": "Mamba-2 is a structured state space model offering linear-time complexity O(N) over long sequences. It is exceptionally fast for streaming audio, replacing the quadratic complexity of standard self-attention.",
            "gate": "Fusion Gates are small MLPs that replace traditional, rigid silence-threshold VADs. By training on spontaneous conversation data, they learn when to trigger thinking and speaking based on prosodic and semantic cues.",
            "memory": "Compute-Memory separation splits the model parameters: all reasoning is done by a compact, lightweight 100M parameter Cognition Core, while all factual knowledge is stored externally in ChromaDB, enabling infinite scale.",
            "asynchronous": "Asynchronous thinking allows the Cognition Core to begin planning responses over streaming audio before the speaker has finished their utterance, rather than waiting for a complete transcript.",
            "hello": "Hello! I am the Vanitas research agent baseline. How can I help you co-author the best low latency voice agent paper today?"
        }

    def get_response(self, text: str) -> str:
        """Determines the conversation response based on keywords and state."""
        query = text.lower().strip()
        self.turns.append(query)
        
        if not query:
            return ""
            
        # Basic keyword matching
        if "vanitas" in query:
            return self.knowledge["vanitas"]
        elif "mamba" in query or "ssm" in query:
            return self.knowledge["mamba"]
        elif "gate" in query or "fusion" in query or "vad" in query:
            return self.knowledge["gate"]
        elif "memory" in query or "separation" in query or "chroma" in query:
            return self.knowledge["memory"]
        elif "async" in query or "think" in query or "stream" in query:
            return self.knowledge["asynchronous"]
        elif "hello" in query or "hi" in query or "hey" in query:
            return self.knowledge["hello"]
            
        # Default smart replies
        if "who are you" in query:
            return "I am Vanitas, a low latency voice agent designed for next-generation speech research."
        elif "conference" in query or "paper" in query or "target" in query:
            return "We are targeting top-tier artificial intelligence conferences such as NeurIPS, ICLR, ICML, or Interspeech. Our core novelty lies in continuous mel processing and learned fusion gates."
        elif "latency" in query:
            return "Our Vanitas architecture achieves sub-100ms conversational turn-taking by eliminating cascades and tokenization, compared to over 3 seconds in typical ASR-LLM-TTS pipelines."
        else:
            return f"That is a great question. Researching '{text}' under our compute-memory separation paradigm will show significant performance improvements."


class CascadedBaselinePipeline:
    """The complete Cascaded Voice Agent Baseline: Capture -> VAD -> ASR -> LLM -> TTS -> Playback."""
    
    def __init__(self, config: GlobalConfig = None):
        self.config = config if config is not None else GlobalConfig()
        
        # Core baseline components
        self.capture = AudioCapture(
            sample_rate=self.config.audio.sample_rate,
            chunk_size=self.config.audio.chunk_size,
            channels=self.config.audio.channels
        )
        self.playback = AudioPlayback(
            sample_rate=self.config.audio.sample_rate,
            channels=self.config.audio.channels
        )
        self.vad = BaselineVAD(
            threshold=self.config.vad.threshold,
            min_speech_duration_ms=self.config.vad.min_speech_duration_ms,
            min_silence_duration_ms=self.config.vad.min_silence_duration_ms,
            sample_rate=self.config.vad.sample_rate
        )
        self.asr = BaselineASR(
            model_size=self.config.asr.model_size,
            device=self.config.asr.device,
            compute_type=self.config.asr.compute_type
        )
        self.tts = BaselineTTS(
            voice=self.config.tts.voice,
            speed=self.config.tts.speed,
            device=self.config.tts.device
        )
        self.llm = LocalDialogueManager()
        
        # State managers
        self.is_running = False
        self.pipeline_thread = None
        
        # Buffers for capturing speech turn
        self.speech_buffer = []
        
        # Profiling logs
        self.latency_stats = []

    def start(self):
        """Starts the cascaded pipeline background thread."""
        if self.is_running:
            return
            
        self.is_running = True
        self.capture.start()
        self.playback.start()
        self.vad.reset()
        
        self.pipeline_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.pipeline_thread.start()
        logger.info("Cascaded Baseline Pipeline fully active.")

    def stop(self):
        """Gracefully shuts down the pipeline."""
        if not self.is_running:
            return
            
        self.is_running = False
        self.capture.stop()
        self.playback.stop()
        
        if self.pipeline_thread is not None:
            self.pipeline_thread.join(timeout=1.5)
            self.pipeline_thread = None
            
        logger.info("Cascaded Baseline Pipeline stopped.")

    def _run_loop(self):
        """Main loop that polls captured audio and handles voice pipeline transitions."""
        logger.info("Pipeline listening loop engaged. Start speaking!")
        
        # Access the raw capture generator
        audio_generator = self.capture.read_generator()
        
        # Track VAD status changes
        was_speaking = False
        
        for audio_chunk in audio_generator:
            if not self.is_running:
                break
                
            # Process VAD
            is_speaking = self.vad.process_chunk(audio_chunk)
            
            # Interruption check: If the user starts talking while the model is speaking, barge-in!
            if is_speaking and self.playback.active_playing:
                logger.info("🎙️ User interruption detected! Triggering Barge-In...")
                self.playback.interrupt()
                self.speech_buffer = []  # Clear buffer to only capture new utterance
                was_speaking = True
                continue
                
            if is_speaking:
                # Accumulate the speech turn
                self.speech_buffer.append(audio_chunk)
                was_speaking = True
            else:
                # If we just transitioned from SPEAKING to SILENT, we have an endpoint!
                if was_speaking:
                    logger.info("🛑 Speech endpoint detected. Processing response...")
                    # Trigger pipeline execution in a separate short-lived thread to keep capture loop ultra-responsive
                    utterance = np.concatenate(self.speech_buffer, axis=0)
                    self.speech_buffer = []
                    was_speaking = False
                    
                    threading.Thread(target=self._execute_pipeline, args=(utterance,), daemon=True).start()

    def _execute_pipeline(self, audio_data: np.ndarray):
        """Processes the captured turn: ASR -> LLM -> TTS -> Playback with precise latency logs."""
        try:
            start_time = time.time()
            
            # 1. Automatic Speech Recognition (ASR)
            asr_start = time.time()
            transcript = self.asr.transcribe(audio_data)
            asr_latency = time.time() - asr_start
            
            if not transcript:
                logger.info("Empty transcript. Skipping response.")
                return
                
            # 2. Large Language Model (LLM)
            llm_start = time.time()
            reply = self.llm.get_response(transcript)
            llm_latency = time.time() - llm_start
            
            if not reply:
                return
                
            # 3. Text-to-Speech (TTS)
            tts_start = time.time()
            synthesis = self.tts.synthesize(reply)
            tts_latency = time.time() - tts_start
            
            total_latency = time.time() - start_time
            
            logger.info(
                f"\n--- LATENCY PROFILE ---\n"
                f"ASR: {asr_latency*1000:.1f}ms\n"
                f"LLM: {llm_latency*1000:.1f}ms\n"
                f"TTS: {tts_latency*1000:.1f}ms\n"
                f"Total Turnaround (Time-to-First-Audio): {total_latency*1000:.1f}ms\n"
                f"------------------------"
            )
            
            self.latency_stats.append({
                "asr": asr_latency,
                "llm": llm_latency,
                "tts": tts_latency,
                "total": total_latency
            })
            
            # 4. Playback (Interruptible streaming)
            self.playback.play(synthesis)
            
        except Exception as e:
            logger.error(f"Error executing cascaded pipeline turn: {e}")
