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
        self.barge_in_threshold = 0.02  # RMS energy to trigger barge-in

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
            config = checkpoint["config"]
            model = VanitasModel(config)
            model.load_state_dict(checkpoint["model_state_dict"])
            
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
        
        user_mel_history = None
        session_memory_embeddings = torch.zeros(1, 0, self.model_config.memory_dim, device=self.device)
        
        # Dialogue loop variables
        agent_speaking = False
        session = {"interrupted": False}
        response_task = None
        
        async def generate_and_stream_response(model_outputs):
            nonlocal agent_speaking
            try:
                audio_tensor = model_outputs["audio"]
                if audio_tensor is None:
                    logger.warning("No audio returned from model inference.")
                    return
                
                audio_np = audio_tensor.detach().cpu().numpy().flatten()
                logger.info(f"Response synthesis complete. Generated {len(audio_np)} samples (~{len(audio_np)/16000:.2f}s). Streaming to client...")
                
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
                            user_mel_history = None
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
                    
                    if agent_speaking and rms > self.barge_in_threshold:
                        logger.info(f"🎙️ User barge-in detected via energy threshold (RMS={rms:.4f})! Interrupting agent...")
                        session["interrupted"] = True
                        if response_task and not response_task.done():
                            response_task.cancel()
                        agent_speaking = False
                        user_mel_history = None
                        session_memory_embeddings = torch.zeros(1, 0, self.model_config.memory_dim, device=self.device)
                        await websocket.send(json.dumps({"type": "interrupted"}))
                        continue
                        
                    # Feed audio to the log-mel extractor
                    new_mel = mel_extractor.feed_audio(pcm_data)
                    if new_mel.size(1) == 0:
                        continue
                        
                    # Append new mel frames to history
                    if user_mel_history is None:
                        user_mel_history = new_mel.to(self.device)
                    else:
                        user_mel_history = torch.cat([user_mel_history, new_mel.to(self.device)], dim=1)
                        
                    # Limit context history to last 10 seconds (1000 frames) to keep operations fast
                    if user_mel_history.size(1) > 1000:
                        user_mel_history = user_mel_history[:, -1000:, :]
                        
                    # Run forward pass of model to calculate gating activations
                    with torch.no_grad():
                        outputs = self.model(
                            user_mel_history, 
                            memory_embeddings=session_memory_embeddings
                        )
                        
                    # Extract last frame activations
                    think_val = outputs["think_gate"][0, -1, 0].item()
                    speak_val = outputs["speak_gate"][0, -1, 0].item()
                    
                    logger.debug(f"Gates: think={think_val:.3f}, speak={speak_val:.3f}")
                    
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
                    if speak_val > self.speak_threshold and not agent_speaking:
                        logger.info(f"🗣️ Speak Gate Fired ({speak_val:.3f}). Initializing generation...")
                        agent_speaking = True
                        session["interrupted"] = False
                        
                        # Start async streaming task
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
