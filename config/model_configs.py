"""
Model configuration classes for TimeGaitNet variants.
"""


class ModelConfig:
    """Base configuration for multitask FOG detection models."""
    
    def __init__(self):
        # Architecture components
        self.use_multiscale_cnn = True
        self.use_vit_encoder = False
        self.use_simple_projection = False
        self.use_temporal_model = True
        self.temporal_model_type = 'lstm'
        self.use_bidirectional = True
        self.use_gated_fusion = True
        self.use_attention_pooling = True
        self.use_channel_attention = True
        
        # Task configuration
        self.task_mode = 'multitask'
        self.sharing_mode = 'hard'
        self.use_uncertainty_loss = False
        self.use_frequency_gating = False
        
        # Model hyperparameters
        self.temporal_hidden = 256
        self.temporal_layers = 2
        self.dropout = 0.3
        self.tcn_kernel_size = 5
        self.tcn_channels = [256, 256, 512]
        
        # Task weights
        self.binary_weight = 0.33
        self.intensity_weight = 0.33
        self.forecast_weight = 0.34


def create_bimamba_nosharing():
    """Standard BiMamba without frequency gating."""
    config = ModelConfig()
    config.temporal_model_type = 'bimamba'
    config.sharing_mode = 'no_sharing'
    config.use_attention_pooling = True
    config.use_channel_attention = True
    config.use_frequency_gating = False
    return config


def create_bimamba_freq_aware():
    """BiMamba with Frequency-Aware Gating."""
    config = ModelConfig()
    config.temporal_model_type = 'bimamba'
    config.sharing_mode = 'no_sharing'
    config.use_attention_pooling = True
    config.use_channel_attention = True
    config.use_frequency_gating = True
    return config
