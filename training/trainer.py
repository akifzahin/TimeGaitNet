"""
Training utilities for TimeGaitNet.
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score, mean_squared_error

from .loss import MultitaskLoss


def train_multitask(model, train_loader, val_loader, epochs=30, device='cuda',
                   dataset_name='', seed=42, config=None, save_dir='./plots'):
    """
    Train multitask FOG detection model.
    
    Args:
        model: MultitaskFOGModel instance
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Maximum number of epochs
        device: Device to train on
        dataset_name: Name for saving checkpoints
        seed: Random seed for reproducibility
        config: ModelConfig instance
        save_dir: Directory for saving plots
        
    Returns:
        history: Dictionary containing training metrics
    """
    patience = 10
    patience_counter = 0
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4)
    from torch.optim.lr_scheduler import LinearLR, SequentialLR
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs-5)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[5])

    # Calculate positive class weight for handling imbalance
    train_binary_pos = 0
    train_total = 0
    for batch in train_loader:
        train_binary_pos += batch['fog_binary'].sum().item()
        train_total += len(batch['fog_binary'])

    pos_weight_binary = torch.tensor([(train_total - train_binary_pos) / train_binary_pos]).to(device) if train_binary_pos > 0 else torch.tensor([1.0]).to(device)
    print(f"Binary pos_weight: {pos_weight_binary.item():.2f}")

    base_criterion = MultitaskLoss(
        binary_weight=config.binary_weight,
        intensity_weight=config.intensity_weight,
        forecast_weight=config.forecast_weight,
        pos_weight=pos_weight_binary, temp=2.0
    )

    history = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
        'train_binary_loss': [], 'val_binary_loss': [], 'train_binary_acc': [], 'val_binary_acc': [],
        'train_intensity_loss': [], 'val_intensity_loss': [], 'train_intensity_acc': [], 'val_intensity_acc': [],
        'train_forecast_loss': [], 'val_forecast_loss': [], 'train_forecast_acc': [], 'val_forecast_acc': [],
        'binary_f1': [], 'binary_prauc': [],
        'intensity_mse': [], 'intensity_mae': [],
        'forecast_f1': [], 'forecast_prauc': []
    }
    best_metric = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_task_losses = {'binary': [], 'intensity': [], 'forecast': []}
        train_binary_preds, train_binary_true = [], []
        train_intensity_preds, train_intensity_true = [], []
        train_forecast_preds, train_forecast_true = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            X = batch['X'].to(device)
            preds = model(X)

            targets = {
                'binary': batch['fog_binary'].float().to(device),
                'intensity': batch['fog_intensity'].float().to(device),
                'forecast': batch['fog_forecast'].float().to(device)
            }

            loss, task_losses = base_criterion(preds, targets)

            for task, task_loss in task_losses.items():
                train_task_losses[task].append(task_loss.item())

            if 'binary' in preds:
                binary_prob = torch.sigmoid(preds['binary']).detach().cpu().numpy()
                train_binary_preds.extend((binary_prob > 0.5).astype(int))
                train_binary_true.extend(batch['fog_binary'].numpy())

            if 'intensity' in preds:
                fog_mask = batch['fog_binary'] == 1
                if fog_mask.sum() > 0:
                    intensity_pred = preds['intensity'].detach().cpu().numpy()[fog_mask.numpy()]
                    train_intensity_preds.extend(intensity_pred)
                    train_intensity_true.extend(batch['fog_intensity'].numpy()[fog_mask.numpy()])

            if 'forecast' in preds:
                forecast_prob = torch.sigmoid(preds['forecast']).detach().cpu().numpy()
                train_forecast_preds.extend((forecast_prob > 0.5).astype(int))
                train_forecast_true.extend(batch['fog_forecast'].numpy())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{np.mean(train_losses):.4f}'})

        scheduler.step()

        # Validation phase
        model.eval()
        val_losses = []
        val_task_losses = {'binary': [], 'intensity': [], 'forecast': []}
        val_binary_preds, val_binary_true, val_binary_probs = [], [], []
        val_intensity_preds, val_intensity_true = [], []
        val_forecast_preds, val_forecast_true, val_forecast_probs = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                X = batch['X'].to(device)
                preds = model(X)

                targets = {
                    'binary': batch['fog_binary'].float().to(device),
                    'intensity': batch['fog_intensity'].float().to(device),
                    'forecast': batch['fog_forecast'].float().to(device)
                }

                loss, task_losses = base_criterion(preds, targets)
                val_losses.append(loss.item())

                for task, task_loss in task_losses.items():
                    val_task_losses[task].append(task_loss.item())

                if 'binary' in preds:
                    binary_prob = torch.sigmoid(preds['binary']).cpu().numpy()
                    val_binary_probs.extend(binary_prob)
                    val_binary_preds.extend((binary_prob > 0.5).astype(int))
                    val_binary_true.extend(batch['fog_binary'].numpy())

                if 'intensity' in preds:
                    fog_mask = batch['fog_binary'] == 1
                    if fog_mask.sum() > 0:
                        val_intensity_preds.extend(preds['intensity'].cpu().numpy()[fog_mask.numpy()])
                        val_intensity_true.extend(batch['fog_intensity'].numpy()[fog_mask.numpy()])

                if 'forecast' in preds:
                    forecast_prob = torch.sigmoid(preds['forecast']).cpu().numpy()
                    val_forecast_probs.extend(forecast_prob)
                    val_forecast_preds.extend((forecast_prob > 0.5).astype(int))
                    val_forecast_true.extend(batch['fog_forecast'].numpy())

        # Compute metrics
        train_binary_acc = np.mean(np.array(train_binary_preds) == np.array(train_binary_true)) if len(train_binary_preds) > 0 else 0
        val_binary_acc = np.mean(np.array(val_binary_preds) == np.array(val_binary_true)) if len(val_binary_preds) > 0 else 0
        train_forecast_acc = np.mean(np.array(train_forecast_preds) == np.array(train_forecast_true)) if len(train_forecast_preds) > 0 else 0
        val_forecast_acc = np.mean(np.array(val_forecast_preds) == np.array(val_forecast_true)) if len(val_forecast_preds) > 0 else 0

        train_intensity_acc = 0
        val_intensity_acc = 0
        if len(train_intensity_preds) > 0:
            train_intensity_acc = np.mean(np.abs(np.array(train_intensity_preds) - np.array(train_intensity_true)) < 0.2)
        if len(val_intensity_preds) > 0:
            val_intensity_acc = np.mean(np.abs(np.array(val_intensity_preds) - np.array(val_intensity_true)) < 0.2)

        train_acc = train_binary_acc
        val_acc = val_binary_acc

        # Binary metrics
        if len(val_binary_preds) > 10:
            binary_probs_clean = np.array(val_binary_probs)
            binary_true_clean = np.array(val_binary_true)
            valid_mask = ~np.isnan(binary_probs_clean) & ~np.isinf(binary_probs_clean)
            if valid_mask.sum() > 10:
                binary_f1 = f1_score(binary_true_clean[valid_mask], np.array(val_binary_preds)[valid_mask], zero_division=0)
                binary_prauc = average_precision_score(binary_true_clean[valid_mask], binary_probs_clean[valid_mask])
            else:
                binary_f1 = 0
                binary_prauc = 0
        else:
            binary_f1 = 0
            binary_prauc = 0

        # Intensity metrics
        if len(val_intensity_preds) > 0:
            intensity_preds_clean = np.array(val_intensity_preds)
            intensity_true_clean = np.array(val_intensity_true)
            valid_mask = ~np.isnan(intensity_preds_clean) & ~np.isinf(intensity_preds_clean)
            if valid_mask.sum() > 0:
                intensity_mse = mean_squared_error(intensity_true_clean[valid_mask], intensity_preds_clean[valid_mask])
                intensity_mae = np.mean(np.abs(intensity_preds_clean[valid_mask] - intensity_true_clean[valid_mask]))
            else:
                intensity_mse = 0.0
                intensity_mae = 0.0
        else:
            intensity_mse = 0.0
            intensity_mae = 0.0

        # Forecast metrics
        if len(val_forecast_preds) > 10:
            forecast_probs_clean = np.array(val_forecast_probs)
            forecast_true_clean = np.array(val_forecast_true)
            valid_mask = ~np.isnan(forecast_probs_clean) & ~np.isinf(forecast_probs_clean)
            if valid_mask.sum() > 10:
                forecast_f1 = f1_score(forecast_true_clean[valid_mask], np.array(val_forecast_preds)[valid_mask], zero_division=0)
                forecast_prauc = average_precision_score(forecast_true_clean[valid_mask], forecast_probs_clean[valid_mask])
            else:
                forecast_f1 = 0
                forecast_prauc = 0
        else:
            forecast_f1 = 0
            forecast_prauc = 0

        # Save to history
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_binary_loss'].append(np.mean(train_task_losses['binary']))
        history['val_binary_loss'].append(np.mean(val_task_losses['binary']))
        history['train_binary_acc'].append(train_binary_acc)
        history['val_binary_acc'].append(val_binary_acc)
        history['train_intensity_loss'].append(np.mean(train_task_losses['intensity']))
        history['val_intensity_loss'].append(np.mean(val_task_losses['intensity']))
        history['train_intensity_acc'].append(train_intensity_acc)
        history['val_intensity_acc'].append(val_intensity_acc)
        history['train_forecast_loss'].append(np.mean(train_task_losses['forecast']))
        history['val_forecast_loss'].append(np.mean(val_task_losses['forecast']))
        history['train_forecast_acc'].append(train_forecast_acc)
        history['val_forecast_acc'].append(val_forecast_acc)
        history['binary_f1'].append(binary_f1)
        history['binary_prauc'].append(binary_prauc)
        history['intensity_mse'].append(intensity_mse)
        history['intensity_mae'].append(intensity_mae)
        history['forecast_f1'].append(forecast_f1)
        history['forecast_prauc'].append(forecast_prauc)

        print(f"Epoch {epoch+1}: Loss {np.mean(train_losses):.4f}/{np.mean(val_losses):.4f} | "
              f"Binary Acc {train_binary_acc:.3f}/{val_binary_acc:.3f} F1 {binary_f1:.3f} | "
              f"Intensity MSE {intensity_mse:.4f} | "
              f"Forecast Acc {train_forecast_acc:.3f}/{val_forecast_acc:.3f} F1 {forecast_f1:.3f}")

        # Combined metric for best model selection
        combined_metric = 0.5 * binary_prauc + 0.3 * (1 - min(intensity_mse, 1.0)) + 0.2 * forecast_prauc

        if combined_metric > best_metric:
            best_metric = combined_metric
            patience_counter = 0
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), f'best_{dataset_name}_seed{seed}_model.pth')
            else:
                torch.save(model.state_dict(), f'best_{dataset_name}_seed{seed}_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return history
