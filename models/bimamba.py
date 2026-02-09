"""
BiMamba encoder implementations for temporal modeling.
"""

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not installed. BiMamba models will not be available.")

from .frequency_aware import FrequencyExtractor, FrequencyAwareGate


class BiMambaEncoder(nn.Module):
    """Bidirectional Mamba encoder for temporal modeling."""
    
    def __init__(self, d_model=256, d_state=16, d_conv=4, expand=2, num_layers=2):
        """
        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            d_conv: Local convolution width
            expand: Expansion factor
            num_layers: Number of Mamba layers
        """
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm not installed")

        self.forward_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])
        self.backward_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])
        self.forward_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.backward_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.output_size = d_model

    def forward(self, x):
        """
        Args:
            x: (B, T, C) input features
            
        Returns:
            (B, T, C) bidirectional features
        """
        x_forward = x
        for layer, norm in zip(self.forward_layers, self.forward_norms):
            residual = x_forward
            x_forward = layer(x_forward)
            if torch.isnan(x_forward).any():
                x_forward = residual
            else:
                x_forward = norm(x_forward + residual)

        x_backward = torch.flip(x, dims=[1])
        for layer, norm in zip(self.backward_layers, self.backward_norms):
            residual = x_backward
            x_backward = layer(x_backward)
            if torch.isnan(x_backward).any():
                x_backward = residual
            else:
                x_backward = norm(x_backward + residual)
        x_backward = torch.flip(x_backward, dims=[1])

        x_combined = torch.cat([x_forward, x_backward], dim=-1)
        return self.fusion(x_combined)


class BiMambaFrequencyAwareEncoder(nn.Module):
    """BiMamba with Frequency-Aware Gating."""
    
    def __init__(self, d_model=256, d_state=16, d_conv=4, expand=2, num_layers=2):
        """
        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            d_conv: Local convolution width
            expand: Expansion factor
            num_layers: Number of Mamba layers
        """
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm not installed")

        # Standard BiMamba components
        self.forward_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])
        self.backward_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])
        self.forward_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.backward_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.fusion = nn.Linear(d_model * 2, d_model)

        # Frequency-aware components
        self.freq_extractor = FrequencyExtractor(d_model)
        self.freq_gate = FrequencyAwareGate(d_model)

        self.output_size = d_model

    def forward(self, x):
        """
        Args:
            x: (B, T, C) input features
            
        Returns:
            (B, T, C) frequency-gated bidirectional features
        """
        # Extract frequency features in parallel
        freq_features = self.freq_extractor(x)  # (B, C)
        gate_weights = self.freq_gate(freq_features)  # (B, C)

        # Standard BiMamba forward pass
        x_forward = x
        for layer, norm in zip(self.forward_layers, self.forward_norms):
            residual = x_forward
            x_forward = layer(x_forward)
            if torch.isnan(x_forward).any():
                x_forward = residual
            else:
                x_forward = norm(x_forward + residual)

        x_backward = torch.flip(x, dims=[1])
        for layer, norm in zip(self.backward_layers, self.backward_norms):
            residual = x_backward
            x_backward = layer(x_backward)
            if torch.isnan(x_backward).any():
                x_backward = residual
            else:
                x_backward = norm(x_backward + residual)
        x_backward = torch.flip(x_backward, dims=[1])

        x_combined = torch.cat([x_forward, x_backward], dim=-1)
        x_fused = self.fusion(x_combined)  # (B, T, C)

        # Apply frequency-aware gating
        # Expand gate_weights to match temporal dimension
        gate_weights_expanded = gate_weights.unsqueeze(1)  # (B, 1, C)
        x_gated = x_fused * gate_weights_expanded  # (B, T, C)

        return x_gated
