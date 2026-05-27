import sys
import time
import click
import signal
import logging
import numpy as np
from vanitas.config import GlobalConfig
from vanitas.pipeline.orchestrator import Orchestrator

# Setup clean visual logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("vanitas.main")

# Global orchestrator reference for clean exit signals
orchestrator = None

def signal_handler(sig, frame):
    """Intercept Ctrl+C and exit gracefully."""
    print("\n")
    logger.info("Interrupt received! Shutting down audio streams gracefully...")
    if orchestrator is not None:
        orchestrator.stop()
        
        # Output latency logs if available
        stats = orchestrator.get_latency_stats()
        if len(stats) > 0:
            print("\n" + "="*35 + " SESSION STATS " + "="*35)
            print(f"{'Turn':<6}{'ASR Latency':<16}{'LLM Latency':<16}{'TTS Latency':<16}{'Total Turnaround (Time-to-Audio)':<30}")
            print("-"*88)
            for idx, turn in enumerate(stats):
                print(f"{idx+1:<6}{turn['asr']*1000:12.1f}ms   {turn['llm']*1000:12.1f}ms   {turn['tts']*1000:12.1f}ms   {turn['total']*1000:24.1f}ms")
            print("="*85 + "\n")
            
    logger.info("Shutdown complete. Goodbye!")
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

@click.command()
@click.option("--mode", default="baseline", type=click.Choice(["baseline", "vanitas"]), help="Mode selection: 'baseline' (ASR-LLM-TTS cascade) vs 'vanitas' (Parallel streams).")
@click.option("--verbose", is_flag=True, help="Print debug logs.")
@click.option("--benchmark", is_flag=True, help="Execute a mock conversation flow to benchmark pipeline turnaround latencies.")
def main(mode, verbose, benchmark):
    """🚀 Vanitas low-latency spoken dialogue agent platform."""
    global orchestrator
    
    # Configure logs
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    print("\n" + "="*35 + " VANITAS PLATFORM " + "="*35)
    print("      A Three-Stream Parallel SSM Voice Agent with Learned Fusion Gates")
    print("="*88 + "\n")
    
    config = GlobalConfig()
    config.verbose = verbose
    
    # Initialize the orchestrator
    orchestrator = Orchestrator(mode=mode, config=config)
    
    if benchmark:
        logger.info("Executing benchmark mode: Simulating speech turns...")
        orchestrator.start()
        
        # Retrieve the underlying baseline pipeline for injecting simulated speech
        pipeline = orchestrator.pipeline
        if pipeline is not None:
            # We inject synthetic voice arrays to evaluate pipeline latencies
            mock_phrases = [
                "Hello, I am testing the turnaround latency of the voice baseline.",
                "Can you explain the difference between Mamba-2 and standard self-attention?",
                "How do learned fusion gates replace Voice Activity Detection?"
            ]
            
            logger.info("Injecting simulated user speech phrases...")
            for idx, phrase in enumerate(mock_phrases):
                print(f"\n--- Benchmark Round {idx+1}: Simulating Speech '{phrase}' ---")
                
                # We mock a transcription directly to evaluate TTS and LLM overheads,
                # or mock a PCM audio array and feed it to pipeline._execute_pipeline
                # Let's feed raw mock audio array of 2.0s
                sample_rate = config.audio.sample_rate
                duration_sec = 2.0
                samples = int(duration_sec * sample_rate)
                # Modulated voice carrier (forces the VAD and ASR baseline fallbacks to process it cleanly)
                t = np.arange(samples)
                pcm = 0.5 * np.sin(200 * 2 * np.pi * t / sample_rate)
                
                # Trigger the pipeline execution synchronously
                pipeline._execute_pipeline(pcm)
                
                # Sleep briefly between turns to simulate natural speaker pacing
                time.sleep(1.5)
                
            # Stop the orchestrator and trigger final signal summary outputs
            orchestrator.stop()
            signal_handler(None, None)
        else:
            logger.error("Failed to run benchmark: orchestrator pipeline not initialized.")
            sys.exit(1)
            
    else:
        # Run normal interactive mode
        orchestrator.start()
        logger.info("Platform active. Keep this process running and speak to your mic!")
        logger.info("Press Ctrl+C to stop the platform and print your session's turnaround latency profile.")
        
        # Infinite sleep loop (signal_handler handles exit)
        while True:
            time.sleep(1.0)

if __name__ == "__main__":
    main()
