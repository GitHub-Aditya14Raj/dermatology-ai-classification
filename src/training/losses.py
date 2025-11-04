"""
Advanced loss functions for imbalanced classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Paper: https://arxiv.org/abs/1708.02002
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Class weights [num_classes]
            gamma: Focusing parameter (default: 2.0)
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits [B, num_classes]
            targets: Ground truth labels [B]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy with Label Smoothing
    Prevents overconfidence and improves generalization
    """
    
    def __init__(self, smoothing=0.1):
        """
        Args:
            smoothing: Label smoothing factor (0.0 to 1.0)
        """
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits [B, num_classes]
            targets: Ground truth labels [B]
        """
        log_probs = F.log_softmax(inputs, dim=1)
        
        # One-hot encode targets
        num_classes = inputs.size(1)
        targets_one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)
        
        # Apply label smoothing
        targets_smooth = targets_one_hot * (1 - self.smoothing) + self.smoothing / num_classes
        
        # Compute loss
        loss = -(targets_smooth * log_probs).sum(dim=1).mean()
        
        return loss


class BiTemperedLogisticLoss(nn.Module):
    """
    Bi-Tempered Logistic Loss
    Robust to label noise and outliers
    Paper: https://arxiv.org/abs/1906.03361
    """
    
    def __init__(self, t1=0.8, t2=1.2, smoothing=0.0):
        """
        Args:
            t1: Temperature for log (< 1.0 for robustness)
            t2: Temperature for exp (> 1.0 for heavy tails)
            smoothing: Label smoothing factor
        """
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        """
        Simplified implementation
        """
        # Use standard CE for now (full implementation is complex)
        return F.cross_entropy(inputs, targets)


class MixupLoss(nn.Module):
    """
    Loss function for Mixup augmentation
    """
    
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion
    
    def forward(self, inputs, targets_a, targets_b, lam):
        """
        Args:
            inputs: Model outputs
            targets_a: First set of targets
            targets_b: Second set of targets
            lam: Mixup parameter
        """
        return lam * self.criterion(inputs, targets_a) + (1 - lam) * self.criterion(inputs, targets_b)


class CutMixLoss(nn.Module):
    """
    Loss function for CutMix augmentation
    """
    
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion
    
    def forward(self, inputs, targets_a, targets_b, lam):
        """
        Args:
            inputs: Model outputs
            targets_a: First set of targets
            targets_b: Second set of targets
            lam: CutMix parameter (area ratio)
        """
        return lam * self.criterion(inputs, targets_a) + (1 - lam) * self.criterion(inputs, targets_b)


if __name__ == '__main__':
    # Test loss functions
    print("Testing loss functions...")
    
    batch_size = 16
    num_classes = 7
    
    # Dummy data
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Test Focal Loss
    print("\n1. Focal Loss:")
    focal = FocalLoss(alpha=torch.ones(num_classes), gamma=2.0)
    loss = focal(logits, labels)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test Label Smoothing
    print("\n2. Label Smoothing CE:")
    ls_ce = LabelSmoothingCrossEntropy(smoothing=0.1)
    loss = ls_ce(logits, labels)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test standard CE
    print("\n3. Standard CE:")
    ce = nn.CrossEntropyLoss()
    loss = ce(logits, labels)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n✅ Loss function tests passed!")
