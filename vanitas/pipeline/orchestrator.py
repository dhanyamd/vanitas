import time
import threading
import asyncio
import logging
from vanitas.config import GlobalConfig
from vanitas.baseline.pipeline import CascadedBaselinePipeline
from vanitas.inference.server import VanitasInferenceServer
from vanitas.inference.client import VanitasTerminalClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vanitas.pipeline.orchestrator")

class VanitasPipelineWrapper:
    """Wraps the Vanitas server and client to run together in interactive mode."""
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.server_thread = None
        self.server = None
        self.client = None
        self.latency_stats = []  # Maintain interface compatibility with main.py stats
        
    def start(self):
        checkpoint_path = self.config.checkpoints_dir / "best_model.pt"
        self.server = VanitasInferenceServer(host="localhost", port=8000, checkpoint_path=str(checkpoint_path))
        
        def run_server_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.server.run())
            
        self.server_thread = threading.Thread(target=run_server_loop, daemon=True)
        self.server_thread.start()
        
        # Allow the server socket to open and bind
        time.sleep(1.5)
        
        # Run client in foreground
        self.client = VanitasTerminalClient(uri="ws://localhost:8000")
        self.client.start()
        
    def stop(self):
        if self.client:
            self.client.stop()

class Orchestrator:
    """The central coordinator that switches modes between baseline (cascaded) and Vanitas (learned)."""
    
    def __init__(self, mode: str = "baseline", config: GlobalConfig = None):
        self.mode = mode.lower()
        self.config = config if config is not None else GlobalConfig()
        
        self.pipeline = None
        
        if self.mode == "baseline":
            logger.info("Initializing Cascaded Voice Agent Baseline Pipeline...")
            self.pipeline = CascadedBaselinePipeline(self.config)
        elif self.mode == "vanitas":
            logger.info("Initializing Vanitas Three-Stream Parallel Pipeline...")
            self.pipeline = VanitasPipelineWrapper(self.config)
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'baseline' or 'vanitas'.")

    def start(self):
        """Starts the active pipeline."""
        if self.pipeline is not None:
            logger.info(f"Starting pipeline in '{self.mode}' mode...")
            self.pipeline.start()
        else:
            logger.warning(
                f"Pipeline is not fully initialized for mode '{self.mode}'. "
                "Phase 2 implementation in progress!"
            )

    def stop(self):
        """Gracefully stops the active pipeline."""
        if self.pipeline is not None:
            logger.info(f"Stopping pipeline in '{self.mode}' mode...")
            self.pipeline.stop()
        else:
            logger.debug("No active pipeline to stop.")
            
    def get_latency_stats(self):
        """Returns the logged latency profiles from the running pipeline."""
        if self.pipeline is not None and hasattr(self.pipeline, "latency_stats"):
            return self.pipeline.latency_stats
        return []
