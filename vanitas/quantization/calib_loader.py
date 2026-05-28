# vanitas/quantization/calib_loader.py
"""Utility to provide a DataLoader of raw audio for calibration.
The loader yields tensors of shape (1, T) containing float32 PCM samples.
It is used by the static PTQ routine to collect activation statistics.
"""
import os
import numpy as np
import soundfile as sf
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

class AudioCalibDataset(Dataset):
    def __init__(self, wav_dir: Path, sample_rate: int = 16000, chunk_ms: int = 20):
        self.paths = list(Path(wav_dir).rglob("*.wav"))
        if not self.paths:
            raise RuntimeError(f"No .wav files found in {wav_dir}")
        self.sr = sample_rate
        self.chunk = int(sample_rate * chunk_ms / 1000)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        wav, _ = sf.read(self.paths[idx], dtype="float32")
        # Ensure at least one chunk length
        if len(wav) < self.chunk:
            wav = np.pad(wav, (0, self.chunk - len(wav)))
        # Trim to a single chunk (good enough for statistics)
        wav = wav[: self.chunk]
        return torch.from_numpy(wav).unsqueeze(0)  # shape (1, chunk)

def get_calib_loader(root_dir: str, batch_size: int = 4, num_workers: int = 2) -> DataLoader:
    ds = AudioCalibDataset(Path(root_dir))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
