"""
Frequency-aware components for FOG detection.
"""

import numpy as np
import torch
import torch.nn as nn


class FrequencyExtractor(nn.Module):
    """Extract frequency features from temporal signals (3-8 Hz FOG band)."""
    
    def __init__(self, d_model, fs=100, window_size=500):
        """
        Args:
            d_model: Model dimension
            fs: Sampling frequency in Hz
            window_size: Window size in samples
        """
        super().__init__()
        self.fs = fs
        self.window_size = window_size

        # Calculate frequency bins for 3-8 Hz
        freqs = np.fft.rfftfreq(window_size, d=1/fs)
        self.fog_band_mask = torch.from_numpy(((freqs >= 3) & (freqs <= 8)).astype(np.float32))
        self.num_freq_bins = self.fog_band_mask.sum().item()

        # Learnable projection for frequency features
        self.freq_projection = nn.Linear(int(self.num_freq_bins), d_model)

    def forward(self, x):
        """
        Args:
            x: (B, T, C) temporal features
            
        Returns:
            (B, C) frequency features
        """
        B, T, C = x.shape

        # Compute FFT along time dimension for each channel
        x_fft = torch.fft.rfft(x, dim=1)  # (B, F, C) where F = freq bins
        x_mag = torch.abs(x_fft)  # Magnitude spectrum

        # Extract FOG-relevant frequency band (3-8 Hz)
        fog_band_mask = self.fog_band_mask.to(x.device)
        fog_band_features = x_mag[:, fog_band_mask.bool(), :]  # (B, num_freq_bins, C)

        # Pool frequency features per channel
        freq_pooled = fog_band_features.mean(dim=1)  # (B, C)

        return freq_pooled


class FrequencyAwareGate(nn.Module):
    """Generate gates based on frequency content."""
    
    def __init__(self, d_model):
        super().__init__()

        self.gate_network = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, freq_features):
        """
        Args:
            freq_features: (B, C) frequency domain features
            
        Returns:
            (B, C) gate weights in [0, 1]
        """
        gate_weights = self.gate_network(freq_features)  # (B, C)
        return gate_weights
