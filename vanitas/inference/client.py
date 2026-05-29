import asyncio
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
import numpy as np
import websockets

# Add project root to python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from vanitas.audio.capture import AudioCapture
from vanitas.audio.playback import AudioPlayback

logging.basicConfig(level=logging.WARNING)  # Mute noisy logs to keep terminal clean
logger = logging.getLogger("vanitas.inference.client")


class VanitasTerminalClient:
    def __init__(self, uri: str = "ws://localhost:8000"):
        self.uri = uri
        
        # Audio configuration
        self.sample_rate = 16000
        self.chunk_size = 512
        self.channels = 1
        
        # Initialize capture and playback
        self.capture = AudioCapture(
            sample_rate=self.sample_rate,
            chunk_size=self.chunk_size,
            channels=self.channels
        )
        self.playback = AudioPlayback(
            sample_rate=self.sample_rate,
            channels=self.channels
        )
        
        # Client states
        self.is_running = False
        self.agent_speaking = False
        self.agent_thinking = False
        self.barge_in_threshold = float(os.environ.get("VANITAS_BARGE_IN_THRESHOLD", "0.20"))
        self.enable_local_barge_in = str(os.environ.get("VANITAS_ENABLE_LOCAL_BARGE_IN", "0")).lower() in {"1", "true", "yes"}
        self.matched_facts = []

    def draw_hud(self, rms_energy: float = 0.0):
        """Draws a premium ASCII dashboard HUD to wow the user."""
        # Clear screen and move cursor to top-left
        sys.stdout.write("\033[H\033[J")
        sys.stdout.write("\033[1;36m" + "="*80 + "\n")
        sys.stdout.write("              🌌  VANITAS LIVE SPONTANEOUS SPEECH COGNITION TERMINAL  🌌\n")
        sys.stdout.write("="*80 + "\033[0m\n\n")
        
        # Display Connection Status
        sys.stdout.write(f"  🔗  Server URI: \033[1;32m{self.uri}\033[0m\n")
        
        # Display Devices Info
        mic_status = "\033[1;32m🟢 Physical Mic\033[0m" if self.capture.has_physical_device else "\033[1;33m🟡 Virtual Simulator\033[0m"
        spk_status = "\033[1;32m🟢 Physical Speaker\033[0m" if self.playback.has_physical_device else "\033[1;33m🟡 Virtual Simulator\033[0m"
        sys.stdout.write(f"  🎙️   Input:  {mic_status}   |   🔊  Output: {spk_status}\n\n")
        
        # Live status lights
        status_text = ""
        if self.agent_speaking:
            status_text = "\033[1;32m🗣️  AGENT SPEAKING [Barge-In Active]\033[0m"
        elif self.agent_thinking:
            status_text = "\033[1;35m🧠  AGENT THINKING (Retrieving Memory...)\033[0m"
        else:
            status_text = "\033[1;34m🎙️  LISTENING (Speak into mic...)\033[0m"
            
        # Draw Energy Bar
        energy_bar_len = int(min(rms_energy * 200, 30))
        energy_bar = "█" * energy_bar_len + "░" * (30 - energy_bar_len)
        sys.stdout.write(f"  [Status] : {status_text}\n")
        sys.stdout.write(f"  [Energy] : \033[1;33m{energy_bar}\033[0m ({rms_energy:.4f})\n\n")
        
        # Draw Retrieved Context facts
        sys.stdout.write("\033[1;34m" + "-"*35 + " COGNITIVE MEMORY RETRIEVAL (RAG) " + "-"*35 + "\033[0m\n")
        if not self.matched_facts:
            sys.stdout.write("  (No vector memory matches retrieved yet. Try asking about 'Vanitas' or 'Mamba'.)\n")
        else:
            for idx, fact in enumerate(self.matched_facts):
                sys.stdout.write(f"  \033[1;33m[{idx+1}]\033[0m {fact}\n")
        sys.stdout.write("\033[1;34m" + "-"*80 + "\033[0m\n\n")
        sys.stdout.write("  \033[1;30mPress Ctrl+C to close dialogue session.\033[0m\n")
        sys.stdout.flush()

    async def send_audio_loop(self, websocket):
        """Continuously captures audio from mic and sends to WebSocket server."""
        self.capture.start()
        audio_gen = self.capture.read_generator()
        
        # Process audio chunks
        for chunk in audio_gen:
            if not self.is_running:
                break
                
            # Compute RMS for energy display
            rms = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0.0
            
            # Draw HUD
            self.draw_hud(rms_energy=rms)
            
            # Send raw PCM float32 bytes
            try:
                await websocket.send(chunk.tobytes())
            except websockets.exceptions.ConnectionClosed:
                break
                
            # If the user speaks while the agent is speaking, trigger local client barge-in instantly!
            if self.enable_local_barge_in and self.agent_speaking and rms > self.barge_in_threshold:
                logger.info("Local barge-in detected. Stopping output stream...")
                self.playback.interrupt()
                self.agent_speaking = False
                # Explicitly notify server of the interruption
                try:
                    await websocket.send(json.dumps({"type": "interrupt"}))
                except Exception:
                    pass
                    
            # Yield to other async tasks
            await asyncio.sleep(0.001)

    async def receive_loop(self, websocket):
        """Receives audio response chunks and control signals from WebSocket server."""
        self.playback.start()
        
        try:
            async for message in websocket:
                # 1. Handle JSON Control Messages
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type")
                        
                        if msg_type == "thinking":
                            self.agent_thinking = True
                            self.matched_facts = data.get("context", [])
                            self.draw_hud()
                            
                        elif msg_type == "speaking_start":
                            self.agent_thinking = False
                            self.agent_speaking = True
                            self.draw_hud()
                            
                        elif msg_type == "speaking_stop":
                            self.agent_speaking = False
                            self.draw_hud()
                            
                        elif msg_type == "interrupted":
                            self.playback.interrupt()
                            self.agent_speaking = False
                            self.agent_thinking = False
                            self.draw_hud()
                            
                    except json.JSONDecodeError:
                        pass
                    continue
                    
                # 2. Handle Binary Audio Stream (Float32 PCM bytes)
                if isinstance(message, bytes):
                    audio_chunk = np.frombuffer(message, dtype=np.float32)
                    self.playback.play(audio_chunk)
                    
        except websockets.exceptions.ConnectionClosed:
            pass

    async def main_loop(self):
        """Establishes connection and runs send/receive tasks concurrently."""
        self.is_running = True
        logger.info(f"Connecting to ws server: {self.uri}...")
        
        try:
            async with websockets.connect(f"{self.uri}/stream") as websocket:
                logger.info("Connected successfully! Starting audio I/O tasks...")
                
                # Run send and receive loops concurrently
                await asyncio.gather(
                    self.send_audio_loop(websocket),
                    self.receive_loop(websocket)
                )
        except Exception as e:
            sys.stdout.write(f"\n❌ Connection failed: {e}\n")
            sys.stdout.flush()
        finally:
            self.stop()

    def start(self):
        """Wrapper to run the async client loop."""
        try:
            asyncio.run(self.main_loop())
        except KeyboardInterrupt:
            sys.stdout.write("\n\n🚪 Terminating dialogue session gracefully...\n")
            sys.stdout.flush()
        finally:
            self.stop()

    def stop(self):
        """Cleans up audio resources."""
        self.is_running = False
        self.capture.stop()
        self.playback.stop()
        sys.stdout.write("🟢 Cleanup completed. Goodbye!\n")
        sys.stdout.flush()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vanitas Live Spoken Dialogue Terminal Client")
    parser.add_argument("--uri", default="ws://localhost:8000", help="WebSocket server address")
    args = parser.parse_args()
    
    client = VanitasTerminalClient(uri=args.uri)
    client.start()
