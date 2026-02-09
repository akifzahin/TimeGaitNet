"""
Helper utilities for TimeGaitNet.
"""

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def get_weighted_sampler(dataset):
    """
    Create weighted sampler for handling class imbalance.
    
    Args:
        dataset: FOGDataset instance
        
    Returns:
        WeightedRandomSampler for DataLoader
    """
    labels = [w['fog_binary'] for w in dataset.windows]
    class_counts = np.bincount(labels, minlength=2)
    weights = 1.0 / (class_counts + 1e-6)
    sample_weights = weights[labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_results_table(results_dict):
    """
    Format results dictionary as readable table.
    
    Args:
        results_dict: Dictionary of metric name -> value
        
    Returns:
        Formatted string table
    """
    lines = ["Metric                     Value"]
    lines.append("-" * 40)
    
    for key, value in results_dict.items():
        if isinstance(value, float):
            lines.append(f"{key:25s} {value:.4f}")
        else:
            lines.append(f"{key:25s} {value}")
    
    return "\n".join(lines)
