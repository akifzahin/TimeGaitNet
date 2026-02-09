"""
Example script demonstrating TimeGaitNet usage.
"""

import torch
from torch.utils.data import DataLoader

from config.model_configs import create_bimamba_freq_aware
from data.dataset import FOGDataset
from models.base_model import MultitaskFOGModel
from training.trainer import train_multitask
from evaluation.evaluator import evaluate_multitask
from utils.helpers import get_weighted_sampler, set_seed, count_parameters


def main():
    """Example training and evaluation pipeline."""
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("\n1. Loading datasets...")
    train_dataset = FOGDataset(
        'path/to/train.parquet',
        window_size=500,
        stride=250,
        augment=True,
        cache_dir='./cache'
    )
    
    test_dataset = FOGDataset(
        'path/to/test.parquet',
        window_size=500,
        stride=250,
        augment=False,
        cache_dir='./cache'
    )
    
    # Create data loaders with weighted sampling
    print("\n2. Creating data loaders...")
    train_sampler = get_weighted_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        sampler=train_sampler,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    print("\n3. Creating model...")
    config = create_bimamba_freq_aware()
    model = MultitaskFOGModel(config)
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Train model
    print("\n4. Training model...")
    history = train_multitask(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=30,
        device=device,
        dataset_name='example',
        seed=42,
        config=config,
        save_dir='./plots'
    )
    
    # Load best model
    print("\n5. Loading best model...")
    model = MultitaskFOGModel(config)
    model.load_state_dict(torch.load('best_example_seed42_model.pth'))
    model = model.to(device)
    
    # Evaluate
    print("\n6. Evaluating model...")
    results = evaluate_multitask(
        model=model,
        loader=test_loader,
        device=device,
        dataset_name='example',
        config=config,
        save_dir='./plots'
    )
    
    # Print results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Binary F1:        {results['binary_f1']:.4f}")
    print(f"Binary PR-AUC:    {results['binary_prauc']:.4f}")
    print(f"Binary ROC-AUC:   {results['binary_roc_auc']:.4f}")
    print(f"Intensity MSE:    {results['intensity_mse']:.4f}")
    print(f"Intensity MAE:    {results['intensity_mae']:.4f}")
    print(f"Forecast F1:      {results['forecast_f1']:.4f}")
    print(f"Forecast PR-AUC:  {results['forecast_prauc']:.4f}")
    print(f"Forecast ROC-AUC: {results['forecast_roc_auc']:.4f}")
    print("="*50)
    
    # Inference example
    print("\n7. Running inference on a single batch...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        X = batch['X'].to(device)
        predictions = model(X)
        
        # Convert logits to probabilities
        binary_probs = torch.sigmoid(predictions['binary'])
        forecast_probs = torch.sigmoid(predictions['forecast'])
        
        print(f"\nSample predictions:")
        print(f"  FOG Detection:  {binary_probs[0]:.3f}")
        print(f"  FOG Intensity:  {predictions['intensity'][0]:.3f}")
        print(f"  FOG Forecast:   {forecast_probs[0]:.3f}")


if __name__ == '__main__':
    main()
