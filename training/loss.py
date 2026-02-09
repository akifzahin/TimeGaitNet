"""
Loss functions for multitask learning.
"""

import numpy as np
import torch
import torch.nn as nn


class MultitaskLoss(nn.Module):
    """
    Adaptive multitask loss with dynamic task weighting.
    
    Balances binary classification, intensity regression, and forecasting
    using relative loss rate adaptation.
    """
    
    def __init__(self, binary_weight=0.33, intensity_weight=0.33, 
                 forecast_weight=0.34, pos_weight=None, temp=2.0):
        """
        Args:
            binary_weight: Base weight for binary task
            intensity_weight: Base weight for intensity task
            forecast_weight: Base weight for forecast task
            pos_weight: Positive class weight for handling class imbalance
            temp: Temperature for adaptive weighting
        """
        super().__init__()
        self.binary_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.forecast_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.intensity_loss = nn.MSELoss()
        self.temp = temp
        self.prev_losses = None
        self.binary_weight = binary_weight
        self.intensity_weight = intensity_weight
        self.forecast_weight = forecast_weight

    def forward(self, preds, targets):
        """
        Args:
            preds: Dictionary with 'binary', 'intensity', 'forecast' predictions
            targets: Dictionary with 'binary', 'intensity', 'forecast' ground truth
            
        Returns:
            total_loss: Weighted sum of task losses
            losses: Dictionary of individual task losses
        """
        losses = {}

        if 'binary' in preds:
            losses['binary'] = self.binary_loss(preds['binary'], targets['binary'])

        if 'intensity' in preds:
            fog_mask = targets['binary'] == 1
            if fog_mask.sum() > 0:
                losses['intensity'] = self.intensity_loss(
                    preds['intensity'][fog_mask],
                    targets['intensity'][fog_mask]
                )
            else:
                losses['intensity'] = torch.tensor(0.0, device=preds['intensity'].device)

        if 'forecast' in preds:
            losses['forecast'] = self.forecast_loss(preds['forecast'], targets['forecast'])

        # Adaptive task weighting based on relative loss rates
        if self.prev_losses is None or any(np.isnan(v) or np.isinf(v) for v in self.prev_losses.values()):
            weights = {
                'binary': self.binary_weight,
                'intensity': self.intensity_weight,
                'forecast': self.forecast_weight
            }
        else:
            try:
                rates = {}
                for k in losses:
                    if k in self.prev_losses and self.prev_losses[k] > 1e-8:
                        rate = losses[k].item() / (self.prev_losses[k] + 1e-8)
                        rates[k] = np.clip(rate, 0.1, 10.0)
                    else:
                        rates[k] = 1.0

                exp_rates = {k: np.exp(np.clip(rates[k] / self.temp, -5, 5)) for k in rates}
                sum_exp = sum(exp_rates.values())

                if sum_exp > 1e-8 and not np.isnan(sum_exp) and not np.isinf(sum_exp):
                    weights = {k: len(rates) * exp_rates[k] / sum_exp for k in rates}
                else:
                    weights = {
                        'binary': self.binary_weight,
                        'intensity': self.intensity_weight,
                        'forecast': self.forecast_weight
                    }
            except:
                weights = {
                    'binary': self.binary_weight,
                    'intensity': self.intensity_weight,
                    'forecast': self.forecast_weight
                }

        self.prev_losses = {k: v.item() for k, v in losses.items() if not torch.isnan(v) and not torch.isinf(v)}

        total_loss = sum(weights.get(k, 0) * v for k, v in losses.items())
        return total_loss, losses
