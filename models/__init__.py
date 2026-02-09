"""
TimeGaitNet models package.
"""

from .base_model import MultitaskFOGModel
from .components import ChannelAttention, MultiScaleCNN, GatedFusion, AttentionPooling
from .bimamba import BiMambaEncoder, BiMambaFrequencyAwareEncoder
from .frequency_aware import FrequencyExtractor, FrequencyAwareGate

__all__ = [
    'MultitaskFOGModel',
    'ChannelAttention',
    'MultiScaleCNN',
    'GatedFusion',
    'AttentionPooling',
    'BiMambaEncoder',
    'BiMambaFrequencyAwareEncoder',
    'FrequencyExtractor',
    'FrequencyAwareGate',
]
