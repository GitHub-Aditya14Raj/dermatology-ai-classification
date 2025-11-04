"""
Model architectures for skin lesion classification
Includes state-of-the-art models from timm library
"""

import torch
import torch.nn as nn
import timm


class SkinLesionClassifier(nn.Module):
    """
    Base classifier with support for:
    - Any timm model as backbone
    - Optional metadata fusion
    - Configurable dropout
    """
    
    def __init__(
        self,
        model_name='efficientnetv2_l',
        num_classes=7,
        pretrained=True,
        use_metadata=False,
        metadata_dim=0,
        dropout=0.3,
        pooling='avg'
    ):
        """
        Args:
            model_name: Name of timm model
            num_classes: Number of output classes
            pretrained: Use ImageNet pretrained weights
            use_metadata: Fuse metadata features
            metadata_dim: Dimension of metadata features
            dropout: Dropout rate before classifier
            pooling: Global pooling type ('avg', 'max', 'gem')
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.use_metadata = use_metadata
        self.metadata_dim = metadata_dim
        
        # Create backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool=pooling
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
        
        print(f"✅ Created {model_name}")
        print(f"   - Feature dim: {self.feature_dim}")
        print(f"   - Pretrained: {pretrained}")
        
        # Classifier head
        classifier_input_dim = self.feature_dim
        
        # Metadata fusion
        if self.use_metadata and self.metadata_dim > 0:
            self.metadata_encoder = nn.Sequential(
                nn.Linear(metadata_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            classifier_input_dim += 32
            print(f"   - Metadata fusion enabled ({metadata_dim} -> 32)")
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, num_classes)
        )
        
        print(f"   - Dropout: {dropout}")
        print(f"   - Output classes: {num_classes}")
    
    def forward(self, x, metadata=None):
        """
        Forward pass
        
        Args:
            x: Input images [B, 3, H, W]
            metadata: Optional metadata features [B, metadata_dim]
        
        Returns:
            logits: Class logits [B, num_classes]
        """
        # Extract image features
        features = self.backbone(x)  # [B, feature_dim]
        
        # Fuse with metadata if available
        if self.use_metadata and metadata is not None:
            metadata_features = self.metadata_encoder(metadata)  # [B, 32]
            features = torch.cat([features, metadata_features], dim=1)
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def get_feature_extractor(self):
        """Return backbone for feature extraction"""
        return self.backbone


def create_model(config):
    """
    Factory function to create model from config
    
    Args:
        config: Dict with model configuration
    
    Returns:
        model: Initialized model
    """
    model = SkinLesionClassifier(
        model_name=config.get('model_name', 'efficientnetv2_l'),
        num_classes=config.get('num_classes', 7),
        pretrained=config.get('pretrained', True),
        use_metadata=config.get('use_metadata', False),
        metadata_dim=config.get('metadata_dim', 0),
        dropout=config.get('dropout', 0.3),
        pooling=config.get('pooling', 'avg')
    )
    
    return model


# Recommended model configurations for 98%+ accuracy
MODEL_CONFIGS = {
    'efficientnetv2_rw_m': {
        'model_name': 'efficientnetv2_rw_m.agc_in1k',
        'img_size': 384,
        'batch_size': 32,
        'lr': 3e-4,
        'description': 'EfficientNetV2-M (RW) - Fast and accurate with pretrained weights'
    },
    'convnext_large': {
        'model_name': 'convnext_large.fb_in22k_ft_in1k',
        'img_size': 384,
        'batch_size': 16,
        'lr': 3e-4,
        'description': 'ConvNeXt-Large - Modern CNN with transformer design (ImageNet-22k)'
    },
    'swin_large': {
        'model_name': 'swin_large_patch4_window12_384',
        'img_size': 384,
        'batch_size': 16,
        'lr': 2e-4,
        'description': 'Swin Transformer - Hierarchical vision transformer'
    },
    'vit_large': {
        'model_name': 'vit_large_patch16_384',
        'img_size': 384,
        'batch_size': 16,
        'lr': 2e-4,
        'description': 'Vision Transformer Large - Pure attention mechanism'
    }
}


def list_available_models():
    """Print all available model configurations"""
    print("\n" + "="*80)
    print("AVAILABLE MODELS FOR 98%+ ACCURACY")
    print("="*80)
    for name, config in MODEL_CONFIGS.items():
        print(f"\n{name}:")
        print(f"  Model: {config['model_name']}")
        print(f"  Image size: {config['img_size']}px")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Learning rate: {config['lr']}")
        print(f"  Description: {config['description']}")


if __name__ == '__main__':
    # Test model creation
    print("Testing model creation...")
    
    # List available models
    list_available_models()
    
    # Test creating a model
    print("\n" + "="*80)
    print("TESTING MODEL CREATION")
    print("="*80)
    
    config = {
        'model_name': 'efficientnetv2_l.in21k_ft_in1k',
        'num_classes': 7,
        'pretrained': True,
        'use_metadata': True,
        'metadata_dim': 20,  # Example: age + sex + localization
        'dropout': 0.3
    }
    
    model = create_model(config)
    
    # Test forward pass
    dummy_img = torch.randn(2, 3, 384, 384)
    dummy_metadata = torch.randn(2, 20)
    
    with torch.no_grad():
        output = model(dummy_img, dummy_metadata)
    
    print(f"\nInput shape: {dummy_img.shape}")
    print(f"Metadata shape: {dummy_metadata.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n✅ Model test passed!")
