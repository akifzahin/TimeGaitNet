"""
Training package for TimeGaitNet.
"""

from .loss import MultitaskLoss
from .trainer import train_multitask

__all__ = ['MultitaskLoss', 'train_multitask']
