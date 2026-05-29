import logging
import os

import torch
import torch.nn as nn
import numpy as np

try:
    from transformers import SpeechT5HifiGan
except Exception:
    SpeechT5HifiGan = None

logger = logging.getLogger("vanitas.model.production.vocoder")


class LightweightNeuralVocoder(nn.Module):
    """Small trainable mel-to-waveform decoder.

    It upsamples 10 ms mel frames by exactly 160 samples at 16 kHz. This is not
    a full HiFi-GAN replacement, but it gives Vanitas a trainable waveform path
    for small base-model fine-tuning and low-latency local inference.
    """

    def __init__(self, mel_dim: int = 80, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(mel_dim, hidden_dim, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, kernel_size=10, stride=5, padding=3, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(hidden_dim // 2, hidden_dim // 4, kernel_size=8, stride=4, padding=2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(hidden_dim // 4, hidden_dim // 8, kernel_size=8, stride=4, padding=2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(hidden_dim // 8, hidden_dim // 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim // 16, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Scale log10 mel range [-5, 2] into a compact range for the decoder.
        x = (torch.clamp(mel, min=-5.0, max=2.0) + 5.0) / 7.0
        x = x.transpose(1, 2)
        return self.net(x)


class HiFiGANWrapper(nn.Module):
    """
    Converts generated log-mel spectrograms to raw waveform audio.

    The default backend is a trainable lightweight neural vocoder. Griffin-Lim
    remains available as a deterministic compatibility fallback, and SpeechT5
    HiFiGAN remains opt-in for diagnostics only.
    """
    def __init__(self, mel_dim: int = 80, hop_length: int = 160):
        super().__init__()
        self.mel_dim = mel_dim
        self.hop_length = hop_length
        self.backend = str(os.environ.get("VANITAS_VOCODER_BACKEND", "neural")).lower()
        self.neural_vocoder = LightweightNeuralVocoder(mel_dim=mel_dim)
        self.vocoder = None
        self._inverse_mel = None
        self._griffin_lim = None

        # HiFiGAN path (fast, but sensitive to mel format mismatch).
        if self.backend in {"hifigan", "auto"}:
            if SpeechT5HifiGan is None:
                logger.warning("transformers SpeechT5HifiGan is unavailable; using Griffin-Lim vocoder.")
            else:
                try:
                    self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
                except Exception as exc:
                    logger.warning("Could not load SpeechT5 HiFiGAN (%s); using Griffin-Lim vocoder.", exc)

        # Griffin-Lim path (format-compatible with our own mel extractor).
        if self.backend in {"griffinlim", "auto"}:
            try:
                import torchaudio.transforms as T
                self._inverse_mel = T.InverseMelScale(
                    n_stft=201,          # n_fft // 2 + 1 for n_fft=400
                    n_mels=self.mel_dim,
                    sample_rate=16000,
                    f_min=0.0,
                    f_max=8000.0,
                    # The default CPU driver ("gels") requires full rank and
                    # fails for this 80-bin/400-FFT mel basis. Rank-tolerant
                    # drivers keep the offline fallback usable.
                    driver="gelsy",
                )
                self._griffin_lim = T.GriffinLim(
                    n_fft=400,
                    hop_length=self.hop_length,
                    win_length=400,
                    n_iter=12,           # balance quality/latency
                    power=2.0,
                )
            except Exception as exc:
                logger.warning("Could not initialize Griffin-Lim vocoder: %s", exc)
                self._inverse_mel = None
                self._griffin_lim = None
        
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, L, mel_dim) - Mel spectrogram
        Returns:
            audio: (B, 1, L * hop_length) - Raw audio waveform
        """
        mel = torch.clamp(mel, min=-5.0, max=2.0)

        if self.backend == "neural":
            return self.neural_vocoder(mel)

        # Clamp to plausible log10 mel range and convert log10 -> natural log.
        mel_ln = mel * np.log(10.0)

        # Prefer Griffin-Lim if available because it matches our mel pipeline.
        if self._inverse_mel is not None and self._griffin_lim is not None:
            try:
                B, L, M = mel.shape
                # Run the inverse on CPU. It is more reliable than MPS for the
                # least-squares inverse and inference immediately copies audio
                # back to CPU for websocket playback anyway.
                mel_power = torch.pow(10.0, mel.detach().float().cpu()).permute(0, 2, 1)  # (B, n_mels, T)
                inverse_mel = self._inverse_mel.cpu()
                griffin_lim = self._griffin_lim.cpu()
                wavs = []
                for b in range(B):
                    spec = inverse_mel(mel_power[b])
                    wav = griffin_lim(torch.clamp(spec, min=1e-8))
                    target_len = L * self.hop_length
                    if wav.numel() < target_len:
                        wav = torch.nn.functional.pad(wav, (0, target_len - wav.numel()))
                    elif wav.numel() > target_len:
                        wav = wav[:target_len]
                    wavs.append(wav)
                audio = torch.stack(wavs, dim=0)  # (B, T)
                return audio.unsqueeze(1).to(mel.device)
            except Exception as exc:
                logger.warning("Griffin-Lim vocoder failed: %s", exc)

        # Fallback to SpeechT5 HiFiGAN when Griffin-Lim backend is unavailable.
        if self.vocoder is None and SpeechT5HifiGan is not None and self.backend in {"hifigan", "auto"}:
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        if self.vocoder is None:
            logger.error("No vocoder backend is available; returning silence instead of crashing inference.")
            return torch.zeros(mel.shape[0], 1, mel.shape[1] * self.hop_length, device=mel.device)

        self.vocoder = self.vocoder.to(mel.device)
        audio = self.vocoder(mel_ln)  # (B, T)
        return audio.unsqueeze(1)
