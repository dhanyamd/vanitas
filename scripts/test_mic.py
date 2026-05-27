import os
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to python path to ensure vanitas imports work cleanly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vanitas.audio.capture import AudioCapture
from vanitas.audio.playback import AudioPlayback

def test_mic_loopback():
    print("="*60)
    print("🎤 VANITAS AUDIO LOOPBACK DIAGNOSTIC TEST 🎤")
    print("="*60)
    
    sample_rate = 16000
    chunk_size = 512
    channels = 1
    
    # Initialize devices
    capture = AudioCapture(sample_rate=sample_rate, chunk_size=chunk_size, channels=channels)
    playback = AudioPlayback(sample_rate=sample_rate, channels=channels)
    
    print("\n[Step 1] Initializing Capture and Playback systems...")
    capture.start()
    playback.start()
    
    # Print status
    if capture.has_physical_device:
        print("🟢 Capture: Physical Microphone detected.")
    else:
        print("🟡 Capture: Virtual soundcard emulator (sine wave generator) engaged.")
        
    if playback.has_physical_device:
        print("🟢 Playback: Physical Speakers detected.")
    else:
        print("🟡 Playback: Virtual soundcard emulator (sleep mode) engaged.")
        
    print("\n[Step 2] Recording audio from capture for 3.0 seconds...")
    print("Speak clearly into your microphone now!")
    print("Recording: ", end="", flush=True)
    
    audio_chunks = []
    start_time = time.time()
    
    # Capture for 3 seconds
    chunk_gen = capture.read_generator()
    for chunk in chunk_gen:
        audio_chunks.append(chunk.copy())
        
        # Calculate energy for simple live feedback
        rms = np.sqrt(np.mean(chunk**2))
        bar = "#" * int(rms * 100)
        print(f"\rRecording: {len(audio_chunks) * chunk_size / sample_rate:.1f}s | Energy: {rms:.4f} {bar:<30}", end="", flush=True)
        
        if time.time() - start_time >= 3.0:
            break
            
    print("\n🟢 Recording complete!")
    capture.stop()
    
    recorded_audio = np.concatenate(audio_chunks, axis=0)
    print(f"Captured {len(recorded_audio)} samples ({len(recorded_audio)/sample_rate:.2f} seconds of audio).")
    
    print("\n[Step 3] Commencing playback loopback...")
    print("You should hear your recorded audio played back...")
    
    playback.play(recorded_audio)
    
    # Wait for playback to complete
    play_duration = len(recorded_audio) / sample_rate
    time.sleep(play_duration + 0.5)
    
    playback.stop()
    print("\n🟢 Playback complete!")
    print("="*60)
    print("🎉 Audio diagnostic completed successfully! 🎉")
    print("="*60)

if __name__ == "__main__":
    test_mic_loopback()
