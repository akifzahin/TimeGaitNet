"""
Run ablation study comparing BiMamba vs BiMamba-FreqAware.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.model_configs import create_bimamba_nosharing, create_bimamba_freq_aware
from data.dataset import FOGDataset
from models.base_model import MultitaskFOGModel
from training.trainer import train_multitask
from evaluation.evaluator import evaluate_multitask
from utils.helpers import get_weighted_sampler, set_seed


def run_multiseed_experiment(config_fn, config_name, dataset_pairs, 
                             num_seeds=5, device='cuda', save_dir='./plots'):
    """
    Run multi-seed experiment for robust evaluation.
    
    Args:
        config_fn: Function that returns ModelConfig
        config_name: Name of configuration
        dataset_pairs: Dictionary of {dataset_name: (train_path, test_path)}
        num_seeds: Number of random seeds to use
        device: Device to train on
        save_dir: Directory for saving plots
        
    Returns:
        df_results: Pandas DataFrame with aggregated results
        all_seed_results: Dictionary with all individual seed results
    """
    print("\n" + "="*70)
    print(f"MULTI-SEED EXPERIMENT: {config_name.upper()} ({num_seeds} seeds)")
    print("="*70)

    all_seed_results = {dataset: {'results': []} for dataset in dataset_pairs.keys()}

    for seed in range(42, 42 + num_seeds):
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)

        for dataset_name, (train_path, test_path) in dataset_pairs.items():
            print(f"\nDataset: {dataset_name.upper()}")

            # Load datasets
            train_dataset = FOGDataset(train_path, augment=True, balance_binary=False)
            test_dataset = FOGDataset(test_path, augment=False, balance_binary=False)

            # Create data loaders
            sampler = get_weighted_sampler(train_dataset)
            train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler, num_workers=2)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

            # Create and train model
            config = config_fn()
            model = MultitaskFOGModel(config)

            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)

            history = train_multitask(
                model, train_loader, test_loader,
                epochs=30, device=device,
                dataset_name=f"{dataset_name}_{config_name}",
                seed=seed,
                config=config,
                save_dir=save_dir
            )

            # Evaluate best model
            model = MultitaskFOGModel(config)
            model.load_state_dict(torch.load(f'best_{dataset_name}_{config_name}_seed{seed}_model.pth'))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(device)

            results = evaluate_multitask(model, test_loader, device=device,
                                        dataset_name=f"{dataset_name}_seed{seed}",
                                        config=config, save_dir=save_dir)

            all_seed_results[dataset_name]['results'].append(results)

    # Aggregate results across seeds
    aggregated_results = []
    for dataset_name, seed_data in all_seed_results.items():
        results_list = seed_data['results']
        all_keys = results_list[0].keys()

        metrics = {}
        for key in all_keys:
            if all(key in r for r in results_list):
                values = [r[key] for r in results_list]
                metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                }

        aggregated_results.append({
            'Dataset': dataset_name,
            'Binary F1': f"{metrics['binary_f1']['mean']:.3f} ± {metrics['binary_f1']['std']:.3f}",
            'Binary PR-AUC': f"{metrics['binary_prauc']['mean']:.3f} ± {metrics['binary_prauc']['std']:.3f}",
            'Binary ROC-AUC': f"{metrics['binary_roc_auc']['mean']:.3f} ± {metrics['binary_roc_auc']['std']:.3f}",
            'Intensity MSE': f"{metrics['intensity_mse']['mean']:.4f} ± {metrics['intensity_mse']['std']:.4f}",
            'Intensity MAE': f"{metrics['intensity_mae']['mean']:.4f} ± {metrics['intensity_mae']['std']:.4f}",
            'Forecast F1': f"{metrics['forecast_f1']['mean']:.3f} ± {metrics['forecast_f1']['std']:.3f}",
            'Forecast PR-AUC': f"{metrics['forecast_prauc']['mean']:.3f} ± {metrics['forecast_prauc']['std']:.3f}",
            'Forecast ROC-AUC': f"{metrics['forecast_roc_auc']['mean']:.3f} ± {metrics['forecast_roc_auc']['std']:.3f}",
        })

    df_results = pd.DataFrame(aggregated_results)
    print("\n" + "="*60)
    print(f"AGGREGATED RESULTS FOR {config_name.upper()} ({num_seeds} seeds)")
    print("="*60)
    print("\n" + df_results.to_string(index=False))

    df_results.to_csv(f'results_{config_name}_multiseed.csv', index=False)

    return df_results, all_seed_results


def main():
    """Main experiment entry point."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Dataset paths - UPDATE THESE TO YOUR ACTUAL PATHS
    tdcsfog_train_path = '/path/to/tdcsfog_train.parquet'
    tdcsfog_test_path = '/path/to/tdcsfog_test.parquet'
    daphnet_train_path = '/path/to/daphnet_train.parquet'
    daphnet_test_path = '/path/to/daphnet_test.parquet'

    dataset_pairs = {
        'daphnet': (daphnet_train_path, daphnet_test_path),
        'tdcsfog': (tdcsfog_train_path, tdcsfog_test_path)
    }

    # Ablation configurations
    ablation_configs = {
        'bimamba_nosharing': create_bimamba_nosharing,
        'bimamba_freq_aware': create_bimamba_freq_aware,
    }

    NUM_SEEDS = 5
    save_dir = './plots'
    os.makedirs(save_dir, exist_ok=True)

    all_results = {}

    # Run experiments
    for ablation_name, config_fn in ablation_configs.items():
        df_results, seed_results = run_multiseed_experiment(
            config_fn, ablation_name, dataset_pairs,
            num_seeds=NUM_SEEDS, device=device, save_dir=save_dir
        )
        all_results[ablation_name] = df_results

    # Print final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON: BiMamba vs BiMamba-FreqAware")
    print("="*80)
    print(f"\n(Results aggregated across {NUM_SEEDS} seeds: mean ± std)")

    for ablation_name, df in all_results.items():
        print(f"\n{ablation_name.upper()}:")
        print(df.to_string(index=False))

    print("\n✓ All results saved to CSV files and plots generated")


if __name__ == '__main__':
    main()
