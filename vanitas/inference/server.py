import asyncio
import json
import logging
import os
import sys
from pathlib import Path
import numpy as np
import torch
import websockets

# Add project root to python path to ensure vanitas imports work
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from vanitas.config import GlobalConfig
from vanitas.model.config import VanitasModelConfig
from vanitas.model.vanitas import VanitasModel
from vanitas.audio.features import MelExtractor
from vanitas.model.cognition.retrieval import FactualRetriever
from vanitas.model.cognition.memory import VectorMemory

# Set up clean logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("vanitas.inference.server")

# Knowledge base to pre-populate Vector DB for testing context retrieval
KNOWLEDGE_BASE = [
    ("Vanitas is our state-of-the-art three-stream parallel neural architecture. It utilizes an always-on Mamba-2 Perception Stream, learned Fusion Gates to eliminate traditional VAD, and a parallel Mamba-2 Production Stream with Flow Matching.", "vanitas_arch"),
    ("Mamba-2 is a structured state space model offering linear-time complexity O(N) over long sequences. It is exceptionally fast for streaming audio, replacing the quadratic complexity of standard self-attention.", "mamba_ssm"),
    ("Fusion Gates are small MLPs that replace traditional, rigid silence-threshold VADs. By training on spontaneous conversation data, they learn when to trigger thinking and speaking based on prosodic and semantic cues.", "fusion_gates"),
    ("Compute-Memory separation splits the model parameters: all reasoning is done by a compact, lightweight 100M parameter Cognition Core, while all factual knowledge is stored externally in ChromaDB, enabling infinite scale.", "compute_memory"),
    ("Asynchronous thinking allows the Cognition Core to begin planning responses over streaming audio before the speaker has finished their utterance, rather than waiting for a complete transcript.", "async_thinking"),
    ("Our Vanitas architecture achieves sub-100ms conversational turn-taking by eliminating cascades and tokenization, compared to over 3 seconds in typical ASR-LLM-TTS pipelines.", "latency_profile")
]


def populate_memory(retriever):
    """Pre-populates the vector memory with dialogue knowledge for RAG retrieval."""
    logger.info("Initializing vector database with factual knowledge base...")
    for text, fact_id in KNOWLEDGE_BASE:
        emb = retriever._deterministic_text_embedding(text).detach().cpu().numpy()
        retriever.memory.add_fact(text, emb, fact_id)
    logger.info(f"Vector Database populated with {len(KNOWLEDGE_BASE)} facts.")


def _config_from_checkpoint(checkpoint: dict) -> VanitasModelConfig:
    raw_config = checkpoint.get("config") or checkpoint.get("model_config")
    if isinstance(raw_config, VanitasModelConfig):
        return raw_config
    if isinstance(raw_config, dict):
        valid_keys = VanitasModelConfig.__dataclass_fields__.keys()
        return VanitasModelConfig(**{k: v for k, v in raw_config.items() if k in valid_keys})
    return VanitasModelConfig()


class VanitasInferenceServer:
    def __init__(self, host: str = "localhost", port: int = 8000, checkpoint_path: str = "checkpoints/best_model.pt",
                 think_threshold: float = 0.45, speak_threshold: float = 0.45):
        self.host = host
        self.port = port
        self.checkpoint_path = checkpoint_path
        
        # Select hardware backend
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        logger.info(f"Using hardware backend: '{self.device}'")
        
        # Load model and configurations
        self.model, self.model_config = self._load_model_checkpoint()
        
        # Initialize Vector Memory and Retriever
        self.memory = VectorMemory(embedding_dim=self.model_config.memory_dim)
        self.retriever = FactualRetriever(self.model_config, memory=self.memory)
        self.retriever.to(self.device)
        self.retriever.eval()
        
        # Pre-populate Vector DB
        populate_memory(self.retriever)
        
        # Settings for Gating
        self.think_threshold = think_threshold
        self.speak_threshold = speak_threshold
        self.barge_in_threshold = 0.15  # RMS energy to trigger barge-in
        # Runtime guards to avoid micro-utterances and choppy retriggers.
        self.min_context_frames = 40          # ~400ms at 10ms/frame
        self.min_response_samples = 3200      # ~200ms at 16kHz
        self.speak_cooldown_s = 0.35
        # Fallback VAD turn-end trigger when learned speak gate is too conservative.
        self.voice_activity_threshold = 0.015
        self.end_of_turn_silence_s = 0.45

    def _prepare_generated_audio(self, audio_np: np.ndarray) -> np.ndarray:
        """Trim leading/trailing near-silence and normalize neural audio for playback."""
        audio_np = np.nan_to_num(audio_np.astype(np.float32, copy=False))
        if audio_np.size == 0:
            return audio_np

        peak = float(np.max(np.abs(audio_np)))
        rms = float(np.sqrt(np.mean(audio_np**2)))
        logger.info(f"Generated audio stats before trim: samples={len(audio_np)}, rms={rms:.6f}, peak={peak:.6f}")
        if peak < 1e-6:
            return np.zeros(0, dtype=np.float32)

        frame = 320
        threshold = max(0.002, 0.08 * peak)
        active = []
        for start in range(0, len(audio_np), frame):
            chunk = audio_np[start:start + frame]
            active.append(float(np.sqrt(np.mean(chunk**2))) > threshold if len(chunk) else False)

        if any(active):
            first = active.index(True) * frame
            last = (len(active) - 1 - active[::-1].index(True) + 1) * frame
            pad = int(0.05 * self.model_config.sample_rate)
            audio_np = audio_np[max(0, first - pad):min(len(audio_np), last + pad)]

        peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
        if peak > 1e-6:
            audio_np = np.clip(audio_np * min(8.0, 0.85 / peak), -1.0, 1.0)
        return audio_np.astype(np.float32, copy=False)

    def _load_model_checkpoint(self):
        """Loads VanitasModel state dict and config from disk."""
        path = Path(self.checkpoint_path)
        if not path.exists():
            logger.error(f"Checkpoint not found at '{path}'. Constructing random initialized model for skeleton testing...")
            config = VanitasModelConfig(
                perception_dim=128,
                cognition_dim=128,
                production_dim=128,
                memory_dim=128,
                mel_bins=80,
                perception_layers=2,
                cognition_layers=2,
                production_layers=2,
                gate_hidden_dim=64
            )
            model = VanitasModel(config)
        else:
            logger.info(f"Loading model checkpoint from '{path}'...")
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            config = _config_from_checkpoint(checkpoint)
            model = VanitasModel(config)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            old_dummy_vocoder = any(k.startswith("vocoder.dummy_conv") for k in state_dict)
            missing_corrected_flow = any(k.startswith("flow_head.mel_mlp") for k in missing)
            if old_dummy_vocoder or missing_corrected_flow:
                logger.warning(
                    "Checkpoint predates the corrected production/vocoder path. "
                    "It may load, but clear speech should not be expected until corrective retraining is run."
                )
            if missing:
                logger.info("Checkpoint load: %d missing keys initialized from current code.", len(missing))
            if unexpected:
                logger.info("Checkpoint load: %d unused checkpoint keys ignored.", len(unexpected))
            
        model.to(self.device)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model successfully loaded. Active parameters: {total_params:,} (~{total_params/1e6:.2f}M)")
        return model, config

    async def run(self):
        """Starts the WebSocket server."""
        async with websockets.serve(self.handler, self.host, self.port):
            logger.info(f"🚀 Vanitas Inference Server running at ws://{self.host}:{self.port}")
            # Keep server running forever
            await asyncio.Event().wait()

    async def handler(self, websocket):
        """Handles incoming WebSocket connections."""
        logger.info(f"Client connected from {websocket.remote_address}")
        
        # Setup session state
        mel_extractor = MelExtractor(
            sample_rate=self.model_config.sample_rate,
            n_fft=400,
            hop_length=self.model_config.hop_length,
            win_length=400,
            n_mels=self.model_config.mel_bins
        )
        agent_mel_extractor = MelExtractor(
            sample_rate=self.model_config.sample_rate,
            n_fft=400,
            hop_length=self.model_config.hop_length,
            win_length=400,
            n_mels=self.model_config.mel_bins
        )
        
        user_mel_history = None
        agent_mel_history = None
        pending_agent_mel_frames = []
        session_memory_embeddings = torch.zeros(1, 0, self.model_config.memory_dim, device=self.device)
        
        # Dialogue loop variables
        agent_speaking = False
        session = {"interrupted": False}
        response_task = None
        last_speak_time = 0.0
        had_voice_since_last_response = False
        last_voice_time = 0.0
        
        async def generate_and_stream_response(model_outputs):
            nonlocal agent_speaking, pending_agent_mel_frames
            try:
                audio_tensor = model_outputs["audio"]
                if audio_tensor is None:
                    logger.warning("No audio returned from model inference.")
                    return
                
                audio_np = audio_tensor.detach().cpu().numpy().flatten()
                audio_np = self._prepare_generated_audio(audio_np)
                if len(audio_np) < self.min_response_samples:
                    logger.info(
                        f"Skipping micro-response ({len(audio_np)} samples < {self.min_response_samples}). "
                        "Waiting for more context before speaking."
                    )
                    return
                logger.info(f"Response synthesis complete. Generated {len(audio_np)} samples (~{len(audio_np)/16000:.2f}s). Streaming to client...")
                
                # Convert generated audio into agent mel frames
                agent_mel_extractor.reset()
                agent_response_mel = agent_mel_extractor.feed_audio(audio_np).to(self.device) # (1, T_gen, n_mels)
                if agent_response_mel.size(1) > 0:
                    pending_agent_mel_frames = [agent_response_mel[:, t:t+1, :] for t in range(agent_response_mel.size(1))]
                else:
                    pending_agent_mel_frames = []
                
                # Signal speaking start
                await websocket.send(json.dumps({"type": "speaking_start"}))
                
                chunk_size = 2048
                for i in range(0, len(audio_np), chunk_size):
                    if session["interrupted"]:
                        logger.info("Streaming cancelled due to interruption.")
                        break
                    
                    chunk = audio_np[i:i + chunk_size]
                    await websocket.send(chunk.tobytes())
                    # Pace the transmission to real-time playout speed (slightly faster to avoid starvation)
                    await asyncio.sleep(len(chunk) / 16000.0 * 0.95)
                    
                logger.info("Finished streaming response.")
            except Exception as e:
                logger.error(f"Error in streaming response: {e}")
            finally:
                agent_speaking = False
                await websocket.send(json.dumps({"type": "speaking_stop"}))

        try:
            async for message in websocket:
                # 1. Handle JSON command messages (String)
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type")
                        
                        if msg_type == "interrupt":
                            logger.info("Barge-in command received from client. Interrupting...")
                            session["interrupted"] = True
                            if response_task and not response_task.done():
                                response_task.cancel()
                            agent_speaking = False
                            pending_agent_mel_frames.clear()
                            user_mel_history = None
                            agent_mel_history = None
                            session_memory_embeddings = torch.zeros(1, 0, self.model_config.memory_dim, device=self.device)
                            await websocket.send(json.dumps({"type": "interrupted"}))
                            
                    except json.JSONDecodeError:
                        logger.warning(f"Malformed JSON string received: {message}")
                    continue

                # 2. Handle Binary Audio Chunks (Bytes)
                if isinstance(message, bytes):
                    # Check for barge-in based on audio energy if the agent is speaking
                    pcm_data = np.frombuffer(message, dtype=np.float32)
                    if len(pcm_data) == 0:
                        continue
                        
                    rms = np.sqrt(np.mean(pcm_data**2))
                    now = asyncio.get_event_loop().time()

                    # Track simple VAD to provide a robust fallback turn-end trigger.
                    if rms > self.voice_activity_threshold:
                        had_voice_since_last_response = True
                        last_voice_time = now
                    
                    if agent_speaking and rms > self.barge_in_threshold:
                        logger.info(f"🎙️ User barge-in detected via energy threshold (RMS={rms:.4f})! Interrupting agent...")
                        session["interrupted"] = True
                        if response_task and not response_task.done():
                            response_task.cancel()
                        agent_speaking = False
                        pending_agent_mel_frames.clear()
                        user_mel_history = None
                        agent_mel_history = None
                        session_memory_embeddings = torch.zeros(1, 0, self.model_config.memory_dim, device=self.device)
                        await websocket.send(json.dumps({"type": "interrupted"}))
                        continue
                        
                    # Feed audio to the log-mel extractor
                    new_user_mel = mel_extractor.feed_audio(pcm_data)
                    if new_user_mel.size(1) == 0:
                        continue
                        
                    T_new = new_user_mel.size(1)
                    
                    # Construct matching agent mel frames (either popped from pending or silence)
                    agent_frames_list = []
                    for _ in range(T_new):
                        if len(pending_agent_mel_frames) > 0:
                            agent_frames_list.append(pending_agent_mel_frames.pop(0))
                        else:
                            agent_frames_list.append(torch.full((1, 1, self.model_config.mel_bins), -5.0, device=self.device))
                    new_agent_mel = torch.cat(agent_frames_list, dim=1) # (1, T_new, mel_bins)
                    
                    # Append new mel frames to history
                    if user_mel_history is None:
                        user_mel_history = new_user_mel.to(self.device)
                        agent_mel_history = new_agent_mel.to(self.device)
                    else:
                        user_mel_history = torch.cat([user_mel_history, new_user_mel.to(self.device)], dim=1)
                        agent_mel_history = torch.cat([agent_mel_history, new_agent_mel.to(self.device)], dim=1)
                        
                    # Limit context history to last 10 seconds (1000 frames) to keep operations fast
                    if user_mel_history.size(1) > 1000:
                        user_mel_history = user_mel_history[:, -1000:, :]
                        agent_mel_history = agent_mel_history[:, -1000:, :]
                        
                    # Run forward pass of model to calculate gating activations
                    with torch.no_grad():
                        outputs = self.model(
                            user_mel_history, 
                            agent_mel_frames=agent_mel_history,
                            memory_embeddings=session_memory_embeddings
                        )
                        
                    # Extract last frame activations
                    think_val = outputs["think_gate"][0, -1, 0].item()
                    speak_val = outputs["speak_gate"][0, -1, 0].item()
                    
                    logger.debug(f"Gates: think={think_val:.3f}, speak={speak_val:.3f}")
                    
                    # Neural interruption: if agent is active but the neural speak gate drops, stop.
                    if agent_speaking and speak_val < self.speak_threshold:
                        logger.info(f"🗣️ Neural yield detected (speak_gate={speak_val:.3f} < {self.speak_threshold})! Interrupting agent...")
                        session["interrupted"] = True
                        if response_task and not response_task.done():
                            response_task.cancel()
                        agent_speaking = False
                        pending_agent_mel_frames.clear()
                        user_mel_history = None
                        agent_mel_history = None
                        session_memory_embeddings = torch.zeros(1, 0, self.model_config.memory_dim, device=self.device)
                        await websocket.send(json.dumps({"type": "interrupted"}))
                        continue
                    
                    # State triggers:
                    # 1. Think Gate fires: query RAG database
                    if think_val > self.think_threshold and session_memory_embeddings.shape[1] == 0:
                        logger.info(f"🎯 Think Gate Fired ({think_val:.3f}). Querying memory...")
                        perception_state = outputs["perception_state"]
                        
                        # Retrieve context
                        session_memory_embeddings, matched_docs = self.retriever(perception_state, top_k=2)
                        
                        logger.info(f"Memory retrieved: {matched_docs[0]}")
                        await websocket.send(json.dumps({
                            "type": "thinking",
                            "context": matched_docs[0]
                        }))
                        
                    # 2. Speak Gate fires: trigger speech response synthesis
                    has_min_context = user_mel_history is not None and user_mel_history.size(1) >= self.min_context_frames
                    cooldown_ok = (now - last_speak_time) >= self.speak_cooldown_s
                    if speak_val > self.speak_threshold and not agent_speaking and has_min_context and cooldown_ok:
                        logger.info(f"🗣️ Speak Gate Fired ({speak_val:.3f}). Initializing generation...")
                        agent_speaking = True
                        session["interrupted"] = False
                        last_speak_time = now
                        had_voice_since_last_response = False
                        
                        # Start async streaming task
                        response_task = asyncio.create_task(generate_and_stream_response(outputs))
                    elif not agent_speaking and has_min_context and cooldown_ok and had_voice_since_last_response:
                        silence_elapsed = now - last_voice_time
                        if silence_elapsed >= self.end_of_turn_silence_s:
                            logger.info(
                                f"🟡 Fallback turn-end trigger fired (silence={silence_elapsed:.2f}s, "
                                f"speak_gate={speak_val:.3f}). Initializing generation..."
                            )
                            agent_speaking = True
                            session["interrupted"] = False
                            last_speak_time = now
                            had_voice_since_last_response = False
                            response_task = asyncio.create_task(generate_and_stream_response(outputs))
                        
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client connection closed.")
        finally:
            if response_task and not response_task.done():
                response_task.cancel()
            logger.info("Session cleaned up.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vanitas Live Inference WebSocket Server")
    parser.add_argument("--host", default="localhost", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--think-threshold", type=float, default=0.45, help="Think gate threshold")
    parser.add_argument("--speak-threshold", type=float, default=0.45, help="Speak gate threshold")
    args = parser.parse_args()
    
    server = VanitasInferenceServer(
        host=args.host, 
        port=args.port, 
        checkpoint_path=args.checkpoint,
        think_threshold=args.think_threshold,
        speak_threshold=args.speak_threshold
    )
    asyncio.run(server.run())
