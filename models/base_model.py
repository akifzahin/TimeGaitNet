"""
Main multitask FOG detection model.
"""

import torch
import torch.nn as nn

from .components import MultiScaleCNN, GatedFusion, AttentionPooling
from .bimamba import BiMambaEncoder, BiMambaFrequencyAwareEncoder


class MultitaskFOGModel(nn.Module):
    """Multitask model for FOG detection, intensity estimation, and forecasting."""
    
    def __init__(self, config):
        """
        Args:
            config: ModelConfig instance with architecture settings
        """
        super().__init__()
        self.config = config

        # CNN Encoder
        self.scale1 = MultiScaleCNN(3, 64, use_channel_attention=config.use_channel_attention)
        self.scale2 = MultiScaleCNN(64, 128, use_channel_attention=config.use_channel_attention)
        self.scale3 = MultiScaleCNN(128, 256, use_channel_attention=config.use_channel_attention)
        self.pool = nn.MaxPool1d(2)
        cnn_out_dim = 256

        # Temporal Model (BiMamba or BiMamba-FA)
        if config.use_temporal_model and config.temporal_model_type == 'bimamba':
            if config.use_frequency_gating:
                print("Using BiMamba with Frequency-Aware Gating")
                self.temporal = BiMambaFrequencyAwareEncoder(
                    d_model=cnn_out_dim, d_state=32, d_conv=4, expand=2, num_layers=3
                )
            else:
                print("Using standard BiMamba")
                self.temporal = BiMambaEncoder(
                    d_model=cnn_out_dim, d_state=32, d_conv=4, expand=2, num_layers=3
                )
            temporal_out_dim = self.temporal.output_size
        else:
            self.temporal = None
            temporal_out_dim = cnn_out_dim

        # Fusion
        if config.use_gated_fusion and self.temporal is not None:
            self.fusion = GatedFusion(cnn_out_dim, temporal_out_dim)
            fusion_out_dim = temporal_out_dim
        else:
            self.fusion = None
            fusion_out_dim = temporal_out_dim

        # Pooling
        if config.use_attention_pooling:
            self.pooling = AttentionPooling(fusion_out_dim)
        else:
            self.pooling = None

        # Task Heads
        if config.sharing_mode == 'hard':
            self.binary_head = self._make_head(fusion_out_dim, 1)
            self.intensity_head = self._make_head(fusion_out_dim, 1)
            self.forecast_head = self._make_head(fusion_out_dim, 1)
        elif config.sharing_mode == 'no_sharing':
            self.binary_encoder = self._make_encoder(fusion_out_dim, fusion_out_dim)
            self.intensity_encoder = self._make_encoder(fusion_out_dim, fusion_out_dim)
            self.forecast_encoder = self._make_encoder(fusion_out_dim, fusion_out_dim)
            self.binary_head = self._make_head(fusion_out_dim, 1)
            self.intensity_head = self._make_head(fusion_out_dim, 1)
            self.forecast_head = self._make_head(fusion_out_dim, 1)

    def _make_encoder(self, in_dim, out_dim):
        """Create task-specific encoder."""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout)
        )

    def _make_head(self, in_dim, out_dim):
        """Create task-specific prediction head."""
        return nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, T) accelerometer input (vertical, mediolateral, anteroposterior)
            
        Returns:
            Dictionary with 'binary', 'intensity', 'forecast' predictions
        """
        # CNN feature extraction
        x = self.pool(self.scale1(x))
        x = self.pool(self.scale2(x))
        cnn_out = self.pool(self.scale3(x))
        cnn_out = cnn_out.transpose(1, 2)

        # Temporal modeling
        if self.temporal is not None:
            temporal_out = self.temporal(cnn_out)
        else:
            temporal_out = cnn_out

        # Fusion
        if self.fusion is not None:
            fused = self.fusion(cnn_out, temporal_out)
        else:
            fused = temporal_out

        # Pooling
        if self.pooling is not None:
            features, _ = self.pooling(fused)
        else:
            features = fused.mean(dim=1)

        # Task-specific predictions
        output = {}
        if self.config.sharing_mode == 'hard':
            output['binary'] = self.binary_head(features).squeeze(-1)
            output['intensity'] = torch.sigmoid(self.intensity_head(features)).squeeze(-1)
            output['forecast'] = self.forecast_head(features).squeeze(-1)
        elif self.config.sharing_mode == 'no_sharing':
            binary_features = self.binary_encoder(features)
            intensity_features = self.intensity_encoder(features)
            forecast_features = self.forecast_encoder(features)
            output['binary'] = self.binary_head(binary_features).squeeze(-1)
            output['intensity'] = torch.sigmoid(self.intensity_head(intensity_features)).squeeze(-1)
            output['forecast'] = self.forecast_head(forecast_features).squeeze(-1)

        return output
