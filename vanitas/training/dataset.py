import torch
from torch.utils.data import Dataset
import numpy as np
import logging

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

from vanitas.config import GlobalConfig
from vanitas.audio.features import MelExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vanitas.training.dataset")

class SpokenDialogueDataset(Dataset):
    """Loads stereo conversational speech, separates channels, and computes on-the-fly Mel features and turn-taking targets."""
    
    def __init__(self, config: GlobalConfig = None, split: str = "train", max_samples: int = None, use_mock: bool = False):
        self.config = config if config is not None else GlobalConfig()
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
                # We load the dataset (which compiles/downloads in a cached folder)
                ds = load_dataset("kyutai/DailyTalkContiguous", split=self.split)
                
                # Slicing if requested
                num_items = len(ds)
                if self.max_samples is not None:
                    num_items = min(num_items, self.max_samples)
                    
                logger.info(f"Loaded {num_items} conversation tracks successfully.")
                
                for idx in range(num_items):
                    item = ds[idx]
                    # In kyutai/DailyTalkContiguous:
                    # audio contains {"array": [2, N] or [N, 2], "sampling_rate": 16000}
                    audio_info = item["audio"]
                    array = np.array(audio_info["array"])
                    sr = audio_info["sampling_rate"]
                    
                    # Ensure shape is (channels, samples)
                    if array.shape[0] != 2:
                        array = array.T # Transpose to (2, N)
                        
                    self.data_samples.append({
                        "array": array,
                        "sample_rate": sr
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
        # Average conversation duration is 8 seconds
        duration = 8.0
        n_samples = int(duration * sr)
        t = np.arange(n_samples)
        
        for _ in range(num_mock):
            # Alternating speaker turns:
            # User (Left channel) talks: 0.5s - 2.5s and 5.0s - 7.0s
            # Agent (Right channel) talks: 2.7s - 4.8s
            left = np.zeros(n_samples, dtype=np.float32)
            right = np.zeros(n_samples, dtype=np.float32)
            
            # User speech synthesis (harmonics + comfort noise)
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
            self.data_samples.append({
                "array": stereo_track,
                "sample_rate": sr
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
        """
        sample = self.data_samples[idx]
        array = sample["array"]
        sr = sample["sample_rate"]
        
        # Resample to 16kHz if needed
        if sr != self.config.audio.sample_rate:
            try:
                import torchaudio.transforms as T
                # Convert to PyTorch tensor for torchaudio resampler
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
        right_channel = array_resampled[1]  # Agent (acts as target boundary indicator)
        
        # 1. Extract log-mel frames from User channel (Left)
        self.mel_extractor.reset()
        mel_tensor = self.mel_extractor.feed_audio(left_channel) # Shape: (1, T, mel_bins)
        mel_input = mel_tensor.squeeze(0)                      # Shape: (T, mel_bins)
        
        T_frames = mel_input.size(0)
        if T_frames == 0:
            # Fallback for empty frame extractions
            mel_input = torch.zeros(10, self.config.audio.n_mels)
            T_frames = 10
            
        # 2. Programmatically calculate turn boundaries from stereo waveforms
        # We calculate RMS energy in 10ms hop bins to match mel frame alignments perfectly!
        hop_samples = self.config.audio.hop_length
        win_samples = self.config.audio.win_length
        
        turn_target = torch.zeros(T_frames, 1, dtype=torch.float32)
        
        # Extract energy values for alignment
        left_energy = []
        right_energy = []
        for t in range(T_frames):
            start_s = t * hop_samples
            end_s = start_s + win_samples
            
            # Slice audio
            l_slice = left_channel[start_s:end_s]
            r_slice = right_channel[start_s:end_s]
            
            l_rms = np.sqrt(np.mean(l_slice**2)) if len(l_slice) > 0 else 0.0
            r_rms = np.sqrt(np.mean(r_slice**2)) if len(r_slice) > 0 else 0.0
            
            left_energy.append(l_rms)
            right_energy.append(r_rms)
            
        # Speech onset/offset binary smoothing
        l_active = np.array(left_energy) > 0.015
        r_active = np.array(right_energy) > 0.015
        
        # Turn-Taking trigger definition:
        # We want our Gate P->C to fire when:
        # User (left) STOPS speaking, AND Agent (right) starts speaking within the next 800ms.
        # This trains the gate to fire exactly at the turn boundary transition!
        for t in range(T_frames - 1):
            # If User was active recently but is now quiet
            user_just_stopped = l_active[t] and not l_active[min(T_frames-1, t+1)]
            
            # If Agent starts speaking in the near future (e.g. within next 40 frames = 400ms)
            agent_starts_soon = False
            lookahead = min(T_frames, t + 40)
            if lookahead > t + 1:
                agent_starts_soon = np.any(r_active[t+1:lookahead])
                
            if user_just_stopped and agent_starts_soon:
                # Mark a 150ms window around the boundary as active (helps train smooth gating)
                start_w = max(0, t - 5)
                end_w = min(T_frames, t + 10)
                turn_target[start_w:end_w, 0] = 1.0

        # 3. Create Masked Mel Spectrogram (15% random masking) for self-supervised pre-training
        masked_mel = mel_input.clone()
        mask_indices = torch.zeros(T_frames, dtype=torch.float32)
        
        # Masking rate of 15%
        num_masked_frames = int(T_frames * 0.15)
        if num_masked_frames > 0:
            # Sample starting indices for masking blocks (consecutive blocks of size 3 = 30ms)
            mask_block_size = 3
            possible_starts = np.arange(T_frames - mask_block_size)
            if len(possible_starts) > 0:
                starts = np.random.choice(possible_starts, size=max(1, num_masked_frames // mask_block_size), replace=False)
                for start in starts:
                    masked_mel[start:start + mask_block_size, :] = 0.0 # Zero out mel features
                    mask_indices[start:start + mask_block_size] = 1.0
                    
        return {
            "mel_input": mel_input,
            "masked_mel": masked_mel,
            "mask_indices": mask_indices,
            "turn_target": turn_target
        }


def pad_collate_fn(batch):
    """Custom collate function to pad variable-length spectrogram sequences."""
    # Find max length in batch
    max_len = max(item["mel_input"].size(0) for item in batch)
    n_mels = batch[0]["mel_input"].size(1)
    
    batch_sz = len(batch)
    
    padded_mel = torch.zeros(batch_sz, max_len, n_mels)
    padded_masked_mel = torch.zeros(batch_sz, max_len, n_mels)
    padded_masks = torch.zeros(batch_sz, max_len)
    padded_turn_targets = torch.zeros(batch_sz, max_len, 1)
    padded_lengths = torch.zeros(batch_sz, dtype=torch.long)
    
    for i, item in enumerate(batch):
        seq_len = item["mel_input"].size(0)
        padded_lengths[i] = seq_len
        
        padded_mel[i, :seq_len, :] = item["mel_input"]
        padded_masked_mel[i, :seq_len, :] = item["masked_mel"]
        padded_masks[i, :seq_len] = item["mask_indices"]
        padded_turn_targets[i, :seq_len, :] = item["turn_target"]
        
    return {
        "mel_input": padded_mel,
        "masked_mel": padded_masked_mel,
        "mask_indices": padded_masks,
        "turn_target": padded_turn_targets,
        "lengths": padded_lengths
    }
