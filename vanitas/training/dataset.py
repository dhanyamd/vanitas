import torch
from torch.utils.data import Dataset
import numpy as np
import logging

import os
import json
import soundfile as sf
from huggingface_hub import hf_hub_download
from vanitas.config import GlobalConfig
from vanitas.model.config import VanitasModelConfig
from vanitas.model.cognition.text_embedding import lexical_text_embedding
from vanitas.audio.features import MelExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vanitas.training.dataset")

def deterministic_text_embedding(text: str, embedding_dim: int = 512) -> torch.Tensor:
    """Dependency-free lexical embedding used for memory alignment targets."""
    return lexical_text_embedding(text, embedding_dim=embedding_dim)

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
        if not self.use_mock:
            try:
                self._load_real_dataset()
            except Exception as e:
                logger.error(f"Failed to load real dataset from HF: {e}")
                import traceback
                traceback.print_exc()
                logger.warning("Falling back to synthetic mock data.")
                self._load_mock_samples()
        else:
            self._load_mock_samples()

    def _load_real_dataset(self):
        """Downloads WAV files from kyutai/DailyTalkContiguous and loads real stereo audio.
        
        Bypasses the HF `datasets` library entirely to avoid CastError on the
        `alignments` column. Instead, downloads the JSONL manifest directly and
        fetches each WAV file individually.
        """
        
        logger.info("Downloading manifest (dailytalk.jsonl) from kyutai/DailyTalkContiguous...")
        
        # Step 1: Download the JSONL manifest directly (no datasets library)
        manifest_path = hf_hub_download(
            repo_id="kyutai/DailyTalkContiguous",
            filename="dailytalk.jsonl",
            repo_type="dataset",
        )
        
        # Step 2: Parse the manifest — each line is {"path": "data_stereo/N.wav", "duration": float, ...}
        manifest = []
        with open(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    manifest.append({"path": entry["path"], "duration": entry["duration"]})
        
        total = len(manifest)
        logger.info(f"Found {total} conversation tracks in manifest.")
        
        # Step 3: Split — first 90% = train, last 10% = val
        split_idx = int(total * 0.9)
        if self.split == "train":
            manifest = manifest[:split_idx]
        else:  # val
            manifest = manifest[split_idx:]
        
        # Apply max_samples limit
        if self.max_samples is not None:
            manifest = manifest[:self.max_samples]
        
        logger.info(f"Downloading {len(manifest)} WAV files for '{self.split}' split...")
        
        # Step 4: Download each WAV and load audio
        loaded = 0
        for idx, entry in enumerate(manifest):
            try:
                wav_path = hf_hub_download(
                    repo_id="kyutai/DailyTalkContiguous",
                    filename=entry["path"],
                    repo_type="dataset",
                )
                
                # Read stereo audio with soundfile
                data, sr = sf.read(wav_path)  # shape: (samples, 2), sr: 44100
                array = data.astype(np.float32)
                
                # Transpose to (channels, samples) = (2, N)
                if array.ndim == 2:
                    array = array.T  # (2, N)
                else:
                    # Mono — duplicate to stereo
                    array = np.stack([array, array], axis=0)
                
                text_context = f"Spoken dialogue conversation {idx}, duration {entry['duration']:.1f}s"
                
                self.data_samples.append({
                    "array": array,
                    "sample_rate": sr,
                    "text_context": text_context,
                })
                loaded += 1
                
                if loaded % 100 == 0:
                    logger.info(f"  Loaded {loaded}/{len(manifest)} tracks...")
                    
            except Exception as e:
                logger.warning(f"  Skipping track {entry['path']}: {e}")
                continue
        
        logger.info(f"✅ Successfully loaded {loaded} REAL conversation tracks for '{self.split}' split.")

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
            
        # Crop sequence length during training to prevent quadratic O(L^2) memory explosion
        max_frames = 256
        if T_frames > max_frames:
            if self.split == "train":
                # Pick a random starting frame
                start_frame = np.random.randint(0, T_frames - max_frames)
            else:
                # Deterministic starting frame for validation
                start_frame = 0
            mel_input = mel_input[start_frame:start_frame + max_frames, :]
            agent_mel = agent_mel[start_frame:start_frame + max_frames, :]
            
            # Slice left_channel and right_channel to match cropped frames for RMS target calculation
            start_s = start_frame * self.config.audio.hop_length
            end_s = start_s + max_frames * self.config.audio.hop_length + self.config.audio.win_length
            left_channel = left_channel[start_s:end_s]
            right_channel = right_channel[start_s:end_s]
            T_frames = max_frames

        target_audio_len = T_frames * self.config.audio.hop_length
        agent_audio = right_channel[:target_audio_len].astype(np.float32)
        if len(agent_audio) < target_audio_len:
            agent_audio = np.pad(agent_audio, (0, target_audio_len - len(agent_audio)))
        agent_audio = torch.from_numpy(agent_audio).unsqueeze(0)
            
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
        agent_active = torch.from_numpy(r_active.astype(np.float32)).unsqueeze(-1)
        l_energy_np = np.array(left_energy, dtype=np.float32)
        r_energy_np = np.array(right_energy, dtype=np.float32)
        
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

        overlap_ratio = float(np.mean(l_active & r_active)) if T_frames else 0.0
        user_active_ratio = float(np.mean(l_active)) if T_frames else 0.0
        agent_active_ratio = float(np.mean(r_active)) if T_frames else 0.0
        prosody_features = torch.tensor(
            [
                float(l_energy_np.mean()),
                float(l_energy_np.std()),
                float(r_energy_np.mean()),
                float(r_energy_np.std()),
                user_active_ratio,
                agent_active_ratio,
                overlap_ratio,
                float(turn_target.mean().item()),
            ],
            dtype=torch.float32,
        )

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
            "agent_active": agent_active,
            "agent_mel": agent_mel,
            "agent_audio": agent_audio,
            "prosody_features": prosody_features,
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
    padded_agent_active = torch.zeros(batch_sz, max_len, 1)
    padded_agent_mel = torch.zeros(batch_sz, max_len, n_mels)
    padded_agent_audio = torch.zeros(batch_sz, 1, max_len * 160)
    padded_prosody_features = torch.zeros(batch_sz, 8)
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
        padded_agent_active[i, :seq_len, :] = item["agent_active"]
        padded_agent_mel[i, :seq_len, :] = item["agent_mel"]
        audio_len = min(item["agent_audio"].size(-1), padded_agent_audio.size(-1))
        padded_agent_audio[i, :, :audio_len] = item["agent_audio"][:, :audio_len]
        padded_prosody_features[i] = item["prosody_features"]
        padded_semantic_targets[i] = item["semantic_target"]
        text_contexts.append(item["text_context"])
        
    return {
        "mel_input": padded_mel,
        "masked_mel": padded_masked_mel,
        "mask_indices": padded_masks,
        "turn_target": padded_turn_targets,
        "agent_active": padded_agent_active,
        "agent_mel": padded_agent_mel,
        "agent_audio": padded_agent_audio,
        "prosody_features": padded_prosody_features,
        "semantic_target": padded_semantic_targets,
        "text_contexts": text_contexts,
        "lengths": padded_lengths
    }
