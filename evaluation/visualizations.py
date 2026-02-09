"""
Visualization utilities for model evaluation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (precision_recall_curve, roc_curve, roc_auc_score, 
                            confusion_matrix, auc, f1_score)


def plot_task_specific_metrics(y_true, y_pred, y_probs, dataset_name, 
                               task_name='binary', save_dir='./plots'):
    """
    Plot comprehensive metrics for a classification task.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities
        dataset_name: Dataset name for plot title
        task_name: Task name (binary, forecast)
        save_dir: Directory to save plots
        
    Returns:
        best_thresh: Optimal threshold for F1 score
        best_f1: Best F1 score achieved
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Precision-Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)

    axes[0, 0].plot(recall, precision, linewidth=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    axes[0, 0].set_xlabel('Recall', fontsize=11)
    axes[0, 0].set_ylabel('Precision', fontsize=11)
    axes[0, 0].set_title(f'{task_name.upper()} - Precision-Recall Curve', fontsize=12, fontweight='bold')
    axes[0, 0].legend(loc='best')
    axes[0, 0].grid(True, alpha=0.3)

    # ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)

    axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.3)
    axes[0, 1].set_xlabel('False Positive Rate', fontsize=11)
    axes[0, 1].set_ylabel('True Positive Rate', fontsize=11)
    axes[0, 1].set_title(f'{task_name.upper()} - ROC Curve', fontsize=12, fontweight='bold')
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid(True, alpha=0.3)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                ax=axes[1, 0], cbar_kws={'label': 'Proportion'})
    axes[1, 0].set_ylabel('True Label', fontsize=11)
    axes[1, 0].set_xlabel('Predicted Label', fontsize=11)
    axes[1, 0].set_title(f'{task_name.upper()} - Normalized Confusion Matrix', fontsize=12, fontweight='bold')

    # F1 Score vs Threshold
    f1_scores = []
    thresholds = np.linspace(0, 1, 100)
    for thresh in thresholds:
        preds_at_thresh = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_true, preds_at_thresh, zero_division=0)
        f1_scores.append(f1)

    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    axes[1, 1].plot(thresholds, f1_scores, linewidth=2, color='green')
    axes[1, 1].axvline(best_thresh, color='red', linestyle='--',
                       label=f'Best Threshold = {best_thresh:.3f}\nF1 = {best_f1:.3f}')
    axes[1, 1].axvline(0.5, color='gray', linestyle=':', alpha=0.5, label='Default (0.5)')
    axes[1, 1].set_xlabel('Threshold', fontsize=11)
    axes[1, 1].set_ylabel('F1 Score', fontsize=11)
    axes[1, 1].set_title(f'{task_name.upper()} - F1 Score vs Threshold', fontsize=12, fontweight='bold')
    axes[1, 1].legend(loc='best')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'{dataset_name.upper()} - {task_name.upper()} Task Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_{task_name}_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    return best_thresh, best_f1


def plot_intensity_analysis(y_true, y_pred, dataset_name, save_dir='./plots'):
    """
    Plot regression analysis for intensity prediction.
    
    Args:
        y_true: Ground truth intensity values
        y_pred: Predicted intensity values
        dataset_name: Dataset name for plot title
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=30, c='blue', edgecolors='k', linewidth=0.5)
    axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect prediction')
    axes[0].set_xlabel('True Intensity', fontsize=11)
    axes[0].set_ylabel('Predicted Intensity', fontsize=11)
    axes[0].set_title('Intensity Predictions Scatter', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Residual plot
    residuals = np.array(y_pred) - np.array(y_true)
    axes[1].scatter(y_true, residuals, alpha=0.5, s=30, c='red', edgecolors='k', linewidth=0.5)
    axes[1].axhline(0, color='black', linestyle='--', linewidth=2)
    axes[1].set_xlabel('True Intensity', fontsize=11)
    axes[1].set_ylabel('Residuals (Predicted - True)', fontsize=11)
    axes[1].set_title('Residual Plot', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Distribution comparison
    axes[2].hist(y_true, bins=20, alpha=0.5, label='True', color='blue', edgecolor='black')
    axes[2].hist(y_pred, bins=20, alpha=0.5, label='Predicted', color='orange', edgecolor='black')
    axes[2].set_xlabel('Intensity Value', fontsize=11)
    axes[2].set_ylabel('Frequency', fontsize=11)
    axes[2].set_title('Distribution Comparison', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'{dataset_name.upper()} - Intensity Regression Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_intensity_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
