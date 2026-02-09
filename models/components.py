"""
Basic building block components for TimeGaitNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention mechanism."""
    
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = x.mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MultiScaleCNN(nn.Module):
    """Multi-scale CNN with parallel convolution branches."""
    
    def __init__(self, in_channels, out_channels, use_channel_attention=True):
        super().__init__()
        self.use_channel_attention = use_channel_attention

        self.branch_3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU()
        )
        self.branch_7 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU()
        )
        self.branch_15 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=15, padding=7),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU()
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.ReLU()
        )

        if use_channel_attention:
            self.channel_attn = ChannelAttention(out_channels)
        else:
            self.channel_attn = None

        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = torch.cat([self.branch_3(x), self.branch_7(x),
                        self.branch_15(x), self.branch_pool(x)], dim=1)

        if self.use_channel_attention and self.channel_attn is not None:
            out = self.channel_attn(out)

        return out + identity


class GatedFusion(nn.Module):
    """Gated fusion module for combining spatial and temporal features."""
    
    def __init__(self, cnn_dim, lstm_dim):
        super().__init__()
        self.spatial_gate = nn.Sequential(
            nn.Linear(cnn_dim, lstm_dim),
            nn.Sigmoid()
        )
        self.temporal_gate = nn.Sequential(
            nn.Linear(lstm_dim, lstm_dim),
            nn.Sigmoid()
        )
        self.fusion = nn.Linear(lstm_dim * 2, lstm_dim)

    def forward(self, cnn_features, lstm_features):
        s_gate = self.spatial_gate(cnn_features)
        t_gate = self.temporal_gate(lstm_features)

        gated_spatial = cnn_features * s_gate if cnn_features.shape[-1] == lstm_features.shape[-1] else lstm_features * s_gate
        gated_temporal = lstm_features * t_gate

        fused = torch.cat([gated_spatial, gated_temporal], dim=-1)
        return self.fusion(fused)


class AttentionPooling(nn.Module):
    """Attention-based pooling over temporal dimension."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, x):
        attn_weights = self.attention(x)
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = (x * attn_weights).sum(dim=1)
        return pooled, attn_weights
