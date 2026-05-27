import logging
from vanitas.config import GlobalConfig
from vanitas.baseline.pipeline import CascadedBaselinePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vanitas.pipeline.orchestrator")

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
            # We will wire the Vanitas parallel pipeline in Phase 2
            self.pipeline = None
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
