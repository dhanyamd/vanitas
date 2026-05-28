import asyncio
import json
import logging
import subprocess
import time
import sys
from pathlib import Path
import numpy as np
import websockets

# Add project root to python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vanitas.test_live_inference")

async def run_latency_test():
    port = 8088
    uri = f"ws://localhost:{port}/stream"
    
    # 1. Start the server as a background subprocess
    logger.info(f"Starting Vanitas Inference Server on port {port}...")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "vanitas.inference.server", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Give the server a moment to boot and load checkpoint
    logger.info("Waiting for server to load checkpoints...")
    await asyncio.sleep(4.0)
    
    try:
        # 2. Connect client to server
        logger.info(f"Connecting to {uri}...")
        async with websockets.connect(uri) as websocket:
            logger.info("Connected successfully! Commencing simulated audio transmission...")
            
            # Prepare synthetic speech data
            # 2 seconds of talking (150Hz tone), followed by 0.5s of silence
            sample_rate = 16000
            chunk_size = 512
            
            t = np.arange(chunk_size)
            rad_s = 2 * np.pi / sample_rate
            active_chunk = (0.3 * np.sin(150 * rad_s * t)).astype(np.float32)
            silent_chunk = np.random.normal(0, 0.001, chunk_size).astype(np.float32)
            
            # Stream 2.0 seconds of talking
            num_speech_chunks = int(2.0 * sample_rate / chunk_size)
            for _ in range(num_speech_chunks):
                await websocket.send(active_chunk.tobytes())
                await asyncio.sleep(chunk_size / sample_rate)
                
            # Send the boundary trigger silence chunks
            logger.info("Speech finished. Sending silence to trigger endpointing gates...")
            
            # Start timer
            start_time = time.perf_counter()
            first_audio_received = False
            first_audio_time = None
            
            # Keep feeding silence to simulate listening while waiting for response
            num_silence_chunks = int(1.0 * sample_rate / chunk_size)
            
            async def send_silence():
                for _ in range(num_silence_chunks):
                    if first_audio_received:
                        break
                    await websocket.send(silent_chunk.tobytes())
                    await asyncio.sleep(chunk_size / sample_rate)
                    
            async def receive_response():
                nonlocal first_audio_received, first_audio_time
                async for message in websocket:
                    if isinstance(message, bytes):
                        first_audio_time = time.perf_counter()
                        first_audio_received = True
                        break
                    elif isinstance(message, str):
                        # Control message received
                        try:
                            data = json.loads(message)
                            logger.info(f"Server control message: {data}")
                        except Exception:
                            pass
            
            # Run send and receive concurrently
            try:
                await asyncio.gather(
                    send_silence(),
                    asyncio.wait_for(receive_response(), timeout=5.0)
                )
            except asyncio.TimeoutError:
                logger.error("Timed out waiting for response from server.")
                
            # 3. Report Latency results
            print("\n" + "="*60)
            print("🔊 VANITAS LOW-LATENCY REAL-TIME INFERENCE RESULTS 🔊")
            print("="*60)
            
            if first_audio_received and first_audio_time is not None:
                turnaround_ms = (first_audio_time - start_time) * 1000.0
                print(f"🟢 First Audio Packet Received successfully!")
                print(f"⏱️  Time-to-First-Audio (Turnaround Latency): {turnaround_ms:.2f} ms")
                print("-"*60)
                if turnaround_ms < 100.0:
                    print("🏆 SUCCESS: SUB-100ms TURNAROUND TARGET MET!")
                else:
                    print("🟡 Target: Turnaround is responsive, but optimized hardware or fewer Euler steps recommended.")
            else:
                print("❌ FAIL: Did not receive synthesized speech output.")
                
            print("="*60 + "\n")
            
    finally:
        # Terminate server process
        logger.info("Shutting down background server process...")
        server_process.terminate()
        try:
            server_process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            server_process.kill()
        logger.info("Server process terminated.")

if __name__ == "__main__":
    asyncio.run(run_latency_test())
