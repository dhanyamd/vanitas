import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vanitas.config import GlobalConfig
from vanitas.audio.features import MelExtractor

def test_features_latency():
    print("="*60)
    print("⚡ VANITAS MEL-SPECTROGRAM EXTRACTION LATENCY TEST ⚡")
    print("="*60)
    
    config = GlobalConfig()
    extractor = MelExtractor(
        sample_rate=config.audio.sample_rate,
        n_fft=config.audio.n_fft,
        hop_length=config.audio.hop_length,
        win_length=config.audio.win_length,
        n_mels=config.audio.n_mels
    )
    
    # Simulate streaming raw PCM chunks (each is 512 samples = 32ms of audio)
    chunk_size = config.audio.chunk_size
    sample_rate = config.audio.sample_rate
    num_rounds = 100
    
    print(f"Simulating {num_rounds} streaming audio blocks of size {chunk_size} (32ms each)...")
    
    latencies = []
    total_frames = 0
    
    for r in range(num_rounds):
        # Generate random audio
        pcm_chunk = np.random.normal(0, 0.05, chunk_size).astype(np.float32)
        
        # Benchmark feature extraction
        start_t = time.time()
        mel_frames = extractor.feed_audio(pcm_chunk)
        elapsed_ms = (time.time() - start_t) * 1000.0
        
        latencies.append(elapsed_ms)
        total_frames += mel_frames.size(1)
        
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    p99_latency = np.percentile(latencies, 99)
    
    print("\n🟢 DSP Features Extraction Summary:")
    print(f"Total Audio Processed: {num_rounds * chunk_size / sample_rate:.2f} seconds")
    print(f"Total Mel Frames Extracted: {total_frames} (10ms hops)")
    print(f"Average DSP Processing Latency: {avg_latency:.4f} ms")
    print(f"Max DSP Processing Latency: {max_latency:.4f} ms")
    print(f"99th Percentile Latency: {p99_latency:.4f} ms")
    
    if avg_latency < 1.0:
        print("\n🏆 Result: SUB-MILLISECOND FEATURE EXTRACTION ACHIEVED! (Elite turn-taking target)")
    else:
        print("\n🟡 Result: Feature extraction is fast, but optimization is recommended.")
        
    print("="*60)

if __name__ == "__main__":
    test_features_latency()
