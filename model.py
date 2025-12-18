# AMD Detection Model Architecture

import torch
import torch.nn as nn
from torchvision import models
import config


class AMDClassifier(nn.Module):
    """
    AMD Classification model based on pretrained ResNet50
    """
    
    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained ImageNet weights
        """
        super(AMDClassifier, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Get the number of features in the last layer
        num_features = self.backbone.fc.in_features
        
        # Replace the final fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)


class AMDEfficientNet(nn.Module):
    """
    AMD Classification model based on pretrained EfficientNet-B0
    """
    
    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained ImageNet weights
        """
        super(AMDEfficientNet, self).__init__()
        
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Get the number of features in the last layer
        num_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)


class AMDVisionTransformer(nn.Module):
    """
    AMD Classification model based on Vision Transformer (ViT)
    """
    
    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained ImageNet weights
        """
        super(AMDVisionTransformer, self).__init__()
        
        # Load pretrained ViT-B/16
        self.backbone = models.vit_b_16(pretrained=pretrained)
        
        # Get the number of features in the last layer
        num_features = self.backbone.heads.head.in_features
        
        # Replace the head
        self.backbone.heads.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)


def get_model(model_name: str = 'resnet50', num_classes: int = config.NUM_CLASSES, 
              pretrained: bool = True, device: str = 'cuda'):
    """
    Get the specified model
    
    Args:
        model_name: Name of the model architecture ('resnet50', 'efficientnet', 'vit')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        device: Device to load the model on
    
    Returns:
        Model instance
    """
    if model_name == 'resnet50':
        model = AMDClassifier(num_classes, pretrained)
    elif model_name == 'efficientnet':
        model = AMDEfficientNet(num_classes, pretrained)
    elif model_name == 'vit':
        model = AMDVisionTransformer(num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{model_name.upper()} Model Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    return model


if __name__ == "__main__":
    # Test model creation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test ResNet50
    model = get_model('resnet50', device=device)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE).to(device)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
