import torch
from torch.utils.data import Dataset
import numpy as np
import logging
import hashlib

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

from vanitas.config import GlobalConfig
from vanitas.model.config import VanitasModelConfig
from vanitas.audio.features import MelExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vanitas.training.dataset")

def deterministic_text_embedding(text: str, embedding_dim: int = 512) -> torch.Tensor:
    """Deterministically encodes a text string into a float32 vector of size `embedding_dim` using SHA-256.
    Acts as a reliable, dependency-free text encoder fallback.
    """
    hasher = hashlib.sha256(text.encode("utf-8"))
    hash_bytes = hasher.digest()
    
    # Use hash bytes to seed a deterministic random generator
    seed = int.from_bytes(hash_bytes[:4], byteorder="big")
    rng = np.random.default_rng(seed)
    
    # Generate random normal vector
    vector = rng.normal(0.0, 0.1, embedding_dim).astype(np.float32)
    
    # Normalize vector
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
        
    return torch.from_numpy(vector)

class SpokenDialogueDataset(Dataset):
    """Loads stereo conversational speech, separates channels, and computes on-the-fly Mel features and turn-taking targets."""
    
    def __init__(self, config: GlobalConfig = None, model_config: VanitasModelConfig = None, split: str = "train", max_samples: int = None, use_mock: bool = False):
        self.config = config if config is not None else GlobalConfig()
        self.model_config = model_config if model_config is not None else VanitasModelConfig()
        self.split = split
        self.max_samples = max_samples
        self.use_mock = use_mock
        
        # Audio feature extractor
        self.mel_extractor = MelExtractor(
            sample_rate=self.config.audio.sample_rate,
            n_fft=self.config.audio.n_fft,
            hop_length=self.config.audio.hop_length,
            win_length=self.config.audio.win_length,
            n_mels=self.config.audio.n_mels
        )
        
        self.data_samples = []
        
        # Load dataset
        if HF_DATASETS_AVAILABLE and not self.use_mock:
            try:
                logger.info(f"Attempting to download/load 'kyutai/DailyTalkContiguous' ({self.split} split) from Hugging Face...")
                ds = load_dataset("kyutai/DailyTalkContiguous", split=self.split)
                
                # Slicing if requested
                num_items = len(ds)
                if self.max_samples is not None:
                    num_items = min(num_items, self.max_samples)
                    
                logger.info(f"Loaded {num_items} conversation tracks successfully.")
                
                for idx in range(num_items):
                    item = ds[idx]
                    audio_info = item["audio"]
                    array = np.array(audio_info["array"])
                    sr = audio_info["sampling_rate"]
                    
                    # Robust transcript extraction
                    text_context = ""
                    if "text" in item:
                        text_context = item["text"]
                    elif "transcript" in item:
                        text_context = item["transcript"]
                    elif "dialogue" in item:
                        if isinstance(item["dialogue"], list):
                            text_context = " ".join([t.get("content", t.get("text", "")) for t in item["dialogue"]])
                        else:
                            text_context = str(item["dialogue"])
                    else:
                        text_context = f"Spoken dialog conversation transcript for sample {idx}"
                    
                    # Ensure shape is (channels, samples)
                    if array.shape[0] != 2:
                        array = array.T # Transpose to (2, N)
                        
                    self.data_samples.append({
                        "array": array,
                        "sample_rate": sr,
                        "text_context": text_context
                    })
            except Exception as e:
                logger.warning(f"Failed to load dataset from HF ({e}). Falling back to synthetic mock data.")
                self._load_mock_samples()
        else:
            self._load_mock_samples()

    def _load_mock_samples(self):
        """Generates mock stereo spoken conversations for offline local testing."""
        num_mock = 50 if self.split == "train" else 10
        if self.max_samples is not None:
            num_mock = min(num_mock, self.max_samples)
            
        logger.info(f"Generating {num_mock} virtual conversational stereo tracks...")
        
        sr = 16000
        duration = 8.0
        n_samples = int(duration * sr)
        t = np.arange(n_samples)
        
        # Hardcoded semantic target facts to align text-audio representations
        mock_facts = [
            "The user is asking about state space modeling and Mamba architectures.",
            "The user wants to configure real-time audio playback thresholds.",
            "The user is checking system diagnostics and checking latency.",
            "The user is asking about the weather forecast for today.",
            "The user is asking about the conversion rates of currencies."
        ]
        
        for idx in range(num_mock):
            # Alternating speaker turns:
            left = np.zeros(n_samples, dtype=np.float32)
            right = np.zeros(n_samples, dtype=np.float32)
            
            rad_s = 2 * np.pi / sr
            user_wave = 0.3 * np.sin(160 * rad_s * t) + 0.1 * np.sin(320 * rad_s * t)
            agent_wave = 0.3 * np.sin(210 * rad_s * t) + 0.1 * np.sin(420 * rad_s * t)
            
            # Turn boundaries
            left[int(0.5*sr):int(2.5*sr)] = user_wave[int(0.5*sr):int(2.5*sr)]
            left[int(5.0*sr):int(7.0*sr)] = user_wave[int(5.0*sr):int(7.0*sr)]
            
            right[int(2.7*sr):int(4.8*sr)] = agent_wave[int(2.7*sr):int(4.8*sr)]
            
            # Add room comfort noise
            left += np.random.normal(0, 0.005, n_samples)
            right += np.random.normal(0, 0.005, n_samples)
            
            stereo_track = np.stack([left, right], axis=0) # (2, N)
            
            fact_text = mock_facts[idx % len(mock_facts)]
            
            self.data_samples.append({
                "array": stereo_track,
                "sample_rate": sr,
                "text_context": fact_text
            })

    def __len__(self) -> int:
        return len(self.data_samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Processes a conversation track, returning features and targets.
        
        Returns:
            dict containing:
                - "mel_input": Tensor of shape (T, mel_bins) (User mel features)
                - "masked_mel": Tensor of shape (T, mel_bins) (User mel with 15% masked frames)
                - "mask_indices": Binary mask of shape (T,) indicating which frames are masked (1=masked)
                - "turn_target": Binary float tensor of shape (T, 1) indicating ground-truth turn boundaries (1=gate fires)
                - "agent_mel": Tensor of shape (T, mel_bins) (Agent mel features for production matching)
                - "text_context": Raw string context fact
                - "semantic_target": Tensor of shape (D_memory,) representing deterministic target embedding for contrastive learning
        """
        sample = self.data_samples[idx]
        array = sample["array"]
        sr = sample["sample_rate"]
        text_context = sample["text_context"]
        
        # Resample to 16kHz if needed
        if sr != self.config.audio.sample_rate:
            try:
                import torchaudio.transforms as T
                tensor_audio = torch.from_numpy(array).float()
                resampler = T.Resample(orig_freq=sr, new_freq=self.config.audio.sample_rate)
                array_resampled = resampler(tensor_audio).numpy()
            except ImportError:
                from scipy.signal import resample
                num_samples = int(array.shape[1] * self.config.audio.sample_rate / sr)
                array_resampled = resample(array, num_samples, axis=1)
        else:
            array_resampled = array
            
        left_channel = array_resampled[0]   # User (always feeds Perception Stream)
        right_channel = array_resampled[1]  # Agent (acts as production target)
        
        # 1. Extract log-mel frames from User channel (Left)
        self.mel_extractor.reset()
        mel_tensor = self.mel_extractor.feed_audio(left_channel) # Shape: (1, T, mel_bins)
        mel_input = mel_tensor.squeeze(0)                      # Shape: (T, mel_bins)
        
        # 2. Extract log-mel frames from Agent channel (Right)
        self.mel_extractor.reset()
        agent_mel_tensor = self.mel_extractor.feed_audio(right_channel) # Shape: (1, T, mel_bins)
        agent_mel = agent_mel_tensor.squeeze(0)                        # Shape: (T, mel_bins)
        
        T_frames = mel_input.size(0)
        if T_frames == 0:
            mel_input = torch.zeros(10, self.config.audio.n_mels)
            agent_mel = torch.zeros(10, self.config.audio.n_mels)
            T_frames = 10
            
        # Ensure agent_mel and mel_input lengths match perfectly
        if agent_mel.size(0) != T_frames:
            # Pad or truncate agent_mel to match mel_input exactly
            aligned_agent = torch.zeros(T_frames, self.config.audio.n_mels)
            min_len = min(T_frames, agent_mel.size(0))
            aligned_agent[:min_len, :] = agent_mel[:min_len, :]
            agent_mel = aligned_agent
            
        # 3. Calculate target turn boundaries
        hop_samples = self.config.audio.hop_length
        win_samples = self.config.audio.win_length
        
        turn_target = torch.zeros(T_frames, 1, dtype=torch.float32)
        
        left_energy = []
        right_energy = []
        for t in range(T_frames):
            start_s = t * hop_samples
            end_s = start_s + win_samples
            
            l_slice = left_channel[start_s:end_s]
            r_slice = right_channel[start_s:end_s]
            
            l_rms = np.sqrt(np.mean(l_slice**2)) if len(l_slice) > 0 else 0.0
            r_rms = np.sqrt(np.mean(r_slice**2)) if len(r_slice) > 0 else 0.0
            
            left_energy.append(l_rms)
            right_energy.append(r_rms)
            
        l_active = np.array(left_energy) > 0.015
        r_active = np.array(right_energy) > 0.015
        
        for t in range(T_frames - 1):
            user_just_stopped = l_active[t] and not l_active[min(T_frames-1, t+1)]
            agent_starts_soon = False
            lookahead = min(T_frames, t + 40)
            if lookahead > t + 1:
                agent_starts_soon = np.any(r_active[t+1:lookahead])
                
            if user_just_stopped and agent_starts_soon:
                start_w = max(0, t - 5)
                end_w = min(T_frames, t + 10)
                turn_target[start_w:end_w, 0] = 1.0

        # 4. Create Masked Mel Spectrogram
        masked_mel = mel_input.clone()
        mask_indices = torch.zeros(T_frames, dtype=torch.float32)
        
        num_masked_frames = int(T_frames * 0.15)
        if num_masked_frames > 0:
            mask_block_size = 3
            possible_starts = np.arange(T_frames - mask_block_size)
            if len(possible_starts) > 0:
                starts = np.random.choice(possible_starts, size=max(1, num_masked_frames // mask_block_size), replace=False)
                for start in starts:
                    masked_mel[start:start + mask_block_size, :] = 0.0
                    mask_indices[start:start + mask_block_size] = 1.0
                    
        # 5. Retrieve target semantic embedding
        semantic_target = deterministic_text_embedding(text_context, embedding_dim=self.model_config.memory_dim)
                    
        return {
            "mel_input": mel_input,
            "masked_mel": masked_mel,
            "mask_indices": mask_indices,
            "turn_target": turn_target,
            "agent_mel": agent_mel,
            "text_context": text_context,
            "semantic_target": semantic_target
        }

def pad_collate_fn(batch):
    """Custom collate function to pad variable-length spectrogram sequences and batch targets."""
    max_len = max(item["mel_input"].size(0) for item in batch)
    n_mels = batch[0]["mel_input"].size(1)
    d_mem = batch[0]["semantic_target"].size(0)
    
    batch_sz = len(batch)
    
    padded_mel = torch.zeros(batch_sz, max_len, n_mels)
    padded_masked_mel = torch.zeros(batch_sz, max_len, n_mels)
    padded_masks = torch.zeros(batch_sz, max_len)
    padded_turn_targets = torch.zeros(batch_sz, max_len, 1)
    padded_agent_mel = torch.zeros(batch_sz, max_len, n_mels)
    padded_semantic_targets = torch.zeros(batch_sz, d_mem)
    padded_lengths = torch.zeros(batch_sz, dtype=torch.long)
    
    text_contexts = []
    
    for i, item in enumerate(batch):
        seq_len = item["mel_input"].size(0)
        padded_lengths[i] = seq_len
        
        padded_mel[i, :seq_len, :] = item["mel_input"]
        padded_masked_mel[i, :seq_len, :] = item["masked_mel"]
        padded_masks[i, :seq_len] = item["mask_indices"]
        padded_turn_targets[i, :seq_len, :] = item["turn_target"]
        padded_agent_mel[i, :seq_len, :] = item["agent_mel"]
        padded_semantic_targets[i] = item["semantic_target"]
        text_contexts.append(item["text_context"])
        
    return {
        "mel_input": padded_mel,
        "masked_mel": padded_masked_mel,
        "mask_indices": padded_masks,
        "turn_target": padded_turn_targets,
        "agent_mel": padded_agent_mel,
        "semantic_target": padded_semantic_targets,
        "text_contexts": text_contexts,
        "lengths": padded_lengths
    }
