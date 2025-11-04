"""
Comprehensive metrics for skin lesion classification
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(y_true, y_pred, y_prob=None, num_classes=7):
    """
    Compute comprehensive classification metrics
    
    Args:
        y_true: Ground truth labels [N]
        y_pred: Predicted labels [N]
        y_prob: Predicted probabilities [N, num_classes] (optional)
        num_classes: Number of classes
    
    Returns:
        metrics: Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Cohen's Kappa
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    for i in range(num_classes):
        metrics[f'precision_class_{i}'] = precision_per_class[i]
        metrics[f'recall_class_{i}'] = recall_per_class[i]
        metrics[f'f1_class_{i}'] = f1_per_class[i]
    
    # ROC-AUC (if probabilities provided)
    if y_prob is not None:
        try:
            # One-vs-Rest ROC-AUC
            metrics['roc_auc_macro'] = roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr')
            metrics['roc_auc_weighted'] = roc_auc_score(y_true, y_prob, average='weighted', multi_class='ovr')
            
            # Per-class ROC-AUC
            for i in range(num_classes):
                y_true_binary = (y_true == i).astype(int)
                if len(np.unique(y_true_binary)) > 1:  # Only compute if class exists
                    metrics[f'roc_auc_class_{i}'] = roc_auc_score(y_true_binary, y_prob[:, i])
        except:
            pass
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics


def evaluate(y_true, y_pred, y_prob=None):
    """
    Compatibility wrapper for simple evaluation calls.

    Args:
        y_true: Ground truth labels [N]
        y_pred: Predicted labels [N]
        y_prob: Predicted probabilities [N, num_classes] (optional)

    Returns:
        metrics: Dictionary with keys 'accuracy', 'f1_macro', 'precision_macro', 'recall_macro', and optionally others.
    """
    # Ensure numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob_arr = None if y_prob is None else np.asarray(y_prob)

    metrics = compute_metrics(y_true, y_pred, y_prob_arr, num_classes=(y_prob_arr.shape[1] if y_prob_arr is not None else int(max(y_true.max(), y_pred.max())+1)))

    # Return a compact set expected by callers
    summary = {
        'accuracy': metrics.get('accuracy', 0.0),
        'f1_macro': metrics.get('f1_macro', 0.0),
        'precision_macro': metrics.get('precision_macro', 0.0),
        'recall_macro': metrics.get('recall_macro', 0.0),
        'confusion_matrix': metrics.get('confusion_matrix')
    }

    # Include roc_auc_macro if present
    if 'roc_auc_macro' in metrics:
        summary['roc_auc_macro'] = metrics['roc_auc_macro']

    return summary


def print_metrics(metrics, class_names=None):
    """
    Print metrics in a readable format
    
    Args:
        metrics: Dictionary of metrics
        class_names: List of class names
    """
    print("\n" + "="*80)
    print("EVALUATION METRICS")
    print("="*80)
    
    # Overall metrics
    print("\n📊 Overall Metrics:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"  F1-Score (macro):  {metrics['f1_macro']:.4f}")
    print(f"  Cohen's Kappa:     {metrics['cohen_kappa']:.4f}")
    
    if 'roc_auc_macro' in metrics:
        print(f"  ROC-AUC (macro):   {metrics['roc_auc_macro']:.4f}")
    
    # Per-class metrics
    print("\n📈 Per-Class Metrics:")
    num_classes = len([k for k in metrics.keys() if k.startswith('f1_class_')])
    
    print(f"\n{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 61)
    
    for i in range(num_classes):
        class_name = class_names[i] if class_names else f"Class {i}"
        precision = metrics[f'precision_class_{i}']
        recall = metrics[f'recall_class_{i}']
        f1 = metrics[f'f1_class_{i}']
        print(f"{class_name:<25} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix [num_classes, num_classes]
        class_names: List of class names
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 10))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'},
        square=True
    )
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved confusion matrix: {save_path}")
    
    plt.close()


def plot_roc_curves(y_true, y_prob, class_names, save_path=None):
    """
    Plot ROC curves for all classes
    
    Args:
        y_true: Ground truth labels [N]
        y_prob: Predicted probabilities [N, num_classes]
        class_names: List of class names
        save_path: Path to save figure
    """
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle
    
    num_classes = len(class_names)
    
    plt.figure(figsize=(12, 10))
    
    # Colors
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink'])
    
    # Plot ROC curve for each class
    for i, color in zip(range(num_classes), colors):
        y_true_binary = (y_true == i).astype(int)
        
        if len(np.unique(y_true_binary)) < 2:
            continue
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(
            fpr, tpr,
            color=color,
            lw=2,
            label=f'{class_names[i]} (AUC = {roc_auc:.3f})'
        )
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Multi-Class Classification')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved ROC curves: {save_path}")
    
    plt.close()


def plot_class_distribution(y_true, y_pred, class_names, save_path=None):
    """
    Plot class distribution comparison
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure
    """
    from collections import Counter
    
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    
    x = np.arange(len(class_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    true_values = [true_counts.get(i, 0) for i in range(len(class_names))]
    pred_values = [pred_counts.get(i, 0) for i in range(len(class_names))]
    
    ax.bar(x - width/2, true_values, width, label='True', color='steelblue')
    ax.bar(x + width/2, pred_values, width, label='Predicted', color='coral')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution: True vs Predicted')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved class distribution: {save_path}")
    
    plt.close()


if __name__ == '__main__':
    # Test metrics
    print("Testing metrics...")
    
    # Dummy data
    np.random.seed(42)
    n_samples = 1000
    num_classes = 7
    
    y_true = np.random.randint(0, num_classes, n_samples)
    y_pred = np.random.randint(0, num_classes, n_samples)
    y_prob = np.random.rand(n_samples, num_classes)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize
    
    class_names = ['mel', 'nv', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes)
    
    # Print metrics
    print_metrics(metrics, class_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, 'test_cm.png')
    
    # Plot ROC curves
    plot_roc_curves(y_true, y_prob, class_names, 'test_roc.png')
    
    print("\n✅ Metrics test passed!")
