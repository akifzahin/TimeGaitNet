"""
Model evaluation utilities.
"""

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (f1_score, average_precision_score, roc_auc_score,
                            classification_report, mean_squared_error)

from .visualizations import plot_task_specific_metrics, plot_intensity_analysis


def evaluate_multitask(model, loader, device='cuda', dataset_name='', 
                      config=None, save_dir='./plots'):
    """
    Evaluate multitask model on all tasks.
    
    Args:
        model: Trained MultitaskFOGModel
        loader: DataLoader for evaluation
        device: Device to run evaluation on
        dataset_name: Dataset name for plots
        config: ModelConfig instance
        save_dir: Directory to save plots
        
    Returns:
        results: Dictionary containing all evaluation metrics
    """
    model.eval()
    binary_preds, binary_true, binary_probs = [], [], []
    intensity_preds, intensity_true = [], []
    forecast_preds, forecast_true, forecast_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {dataset_name}"):
            X = batch['X'].to(device)
            preds = model(X)

            if 'binary' in preds:
                binary_prob = torch.sigmoid(preds['binary']).cpu().numpy()
                binary_probs.extend(binary_prob)
                binary_preds.extend((binary_prob > 0.5).astype(int))
                binary_true.extend(batch['fog_binary'].numpy())

            if 'intensity' in preds:
                fog_mask = batch['fog_binary'] == 1
                if fog_mask.sum() > 0:
                    intensity_preds.extend(preds['intensity'].cpu().numpy()[fog_mask.numpy()])
                    intensity_true.extend(batch['fog_intensity'].numpy()[fog_mask.numpy()])

            if 'forecast' in preds:
                forecast_prob = torch.sigmoid(preds['forecast']).cpu().numpy()
                forecast_probs.extend(forecast_prob)
                forecast_preds.extend((forecast_prob > 0.5).astype(int))
                forecast_true.extend(batch['fog_forecast'].numpy())

    results = {}

    # Binary FOG Detection Evaluation
    if len(binary_preds) > 10:
        binary_probs_clean = np.array(binary_probs)
        binary_true_clean = np.array(binary_true)
        binary_preds_clean = np.array(binary_preds)
        valid_mask = ~np.isnan(binary_probs_clean) & ~np.isinf(binary_probs_clean)

        if valid_mask.sum() > 10:
            print("\nBinary FOG Detection Report:")
            print(classification_report(binary_true_clean[valid_mask], binary_preds_clean[valid_mask],
                                       labels=[0, 1], target_names=['No FOG', 'FOG'], zero_division=0))
            results['binary_f1'] = f1_score(binary_true_clean[valid_mask], binary_preds_clean[valid_mask], zero_division=0)
            results['binary_prauc'] = average_precision_score(binary_true_clean[valid_mask], binary_probs_clean[valid_mask])
            results['binary_roc_auc'] = roc_auc_score(binary_true_clean[valid_mask], binary_probs_clean[valid_mask])

            best_thresh, best_f1 = plot_task_specific_metrics(
                binary_true_clean[valid_mask], binary_preds_clean[valid_mask], binary_probs_clean[valid_mask],
                dataset_name, 'binary', save_dir
            )
            results['binary_best_threshold'] = best_thresh
            results['binary_best_f1'] = best_f1
        else:
            results['binary_f1'] = 0.0
            results['binary_prauc'] = 0.0
            results['binary_roc_auc'] = 0.0
    else:
        results['binary_f1'] = 0.0
        results['binary_prauc'] = 0.0
        results['binary_roc_auc'] = 0.0

    # Intensity Regression Evaluation
    if len(intensity_preds) > 0:
        intensity_preds_clean = np.array(intensity_preds)
        intensity_true_clean = np.array(intensity_true)
        valid_mask = ~np.isnan(intensity_preds_clean) & ~np.isinf(intensity_preds_clean)

        if valid_mask.sum() > 0:
            plot_intensity_analysis(intensity_true_clean[valid_mask], intensity_preds_clean[valid_mask],
                                   dataset_name, save_dir)
            results['intensity_mse'] = mean_squared_error(intensity_true_clean[valid_mask],
                                                          intensity_preds_clean[valid_mask])
            results['intensity_mae'] = np.mean(np.abs(intensity_preds_clean[valid_mask] -
                                                      intensity_true_clean[valid_mask]))
            print(f"\nIntensity Regression (FOG samples only):")
            print(f"  MSE: {results['intensity_mse']:.4f}")
            print(f"  MAE: {results['intensity_mae']:.4f}")
        else:
            results['intensity_mse'] = 0.0
            results['intensity_mae'] = 0.0
    else:
        results['intensity_mse'] = 0.0
        results['intensity_mae'] = 0.0

    # Forecast Evaluation
    if len(forecast_preds) > 10:
        forecast_probs_clean = np.array(forecast_probs)
        forecast_true_clean = np.array(forecast_true)
        forecast_preds_clean = np.array(forecast_preds)
        valid_mask = ~np.isnan(forecast_probs_clean) & ~np.isinf(forecast_probs_clean)

        if valid_mask.sum() > 10:
            print("\nForecast (5s ahead) Report:")
            print(classification_report(forecast_true_clean[valid_mask], forecast_preds_clean[valid_mask],
                                       labels=[0, 1], target_names=['No FOG upcoming', 'FOG upcoming'],
                                       zero_division=0))
            results['forecast_f1'] = f1_score(forecast_true_clean[valid_mask], forecast_preds_clean[valid_mask],
                                              zero_division=0)
            results['forecast_prauc'] = average_precision_score(forecast_true_clean[valid_mask],
                                                                forecast_probs_clean[valid_mask])
            results['forecast_roc_auc'] = roc_auc_score(forecast_true_clean[valid_mask],
                                                        forecast_probs_clean[valid_mask])

            best_thresh, best_f1 = plot_task_specific_metrics(
                forecast_true_clean[valid_mask], forecast_preds_clean[valid_mask], forecast_probs_clean[valid_mask],
                dataset_name, 'forecast', save_dir
            )
            results['forecast_best_threshold'] = best_thresh
            results['forecast_best_f1'] = best_f1
        else:
            results['forecast_f1'] = 0.0
            results['forecast_prauc'] = 0.0
            results['forecast_roc_auc'] = 0.0
    else:
        results['forecast_f1'] = 0.0
        results['forecast_prauc'] = 0.0
        results['forecast_roc_auc'] = 0.0

    return results
